"""Read verl console logs and run the PTH checks on them.

Works with the console output of ``python -m verl.trainer.main_ppo`` (and
entry points built on it) with ``trainer.logger`` including ``console``.
verl prints the resolved configuration at start-up and one
``step:N - key:value - ...`` line per update. The config layout follows verl
releases that have ``algorithm.rollout_correction`` (late 2025).
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, Mapping, Optional

from .report import Report
from .ratio import scan_clip_vs_kl

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RAY_PREFIX = re.compile(r"^(\([A-Za-z_][\w]* pid=\d+(?:, ip=[^)]*)?\)) ?(.*)$")
_CONTINUATION = re.compile(r"""^\s*(['"\[\{\(\-\d]|True\b|False\b|None\b)""")
_STEP = re.compile(r"step:(\d+) - (.*)")

CLIP_KEY = "actor/pg_clipfrac"
KL_KEY = "rollout_corr/kl"
IS_KEY = "rollout_corr/rollout_is_mean"


def _lines(path: str):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            yield _ANSI.sub("", raw.rstrip("\n"))


def read_metrics(path: str) -> Dict[int, Dict[str, float]]:
    """Parse every ``step:N - key:value - ...`` line into ``{N: {key: value}}``."""
    out: Dict[int, Dict[str, float]] = {}
    for line in _lines(path):
        m = _STEP.search(line)
        if not m:
            continue
        row: Dict[str, float] = {}
        for item in m.group(2).split(" - "):
            key, sep, val = item.strip().rpartition(":")
            if not sep:
                continue
            try:
                row[key] = float(val)
            except ValueError:
                continue
        out.setdefault(int(m.group(1)), {}).update(row)
    return out


def read_config(path: str) -> Optional[Dict[str, Any]]:
    """Recover the resolved configuration verl prints at start-up.

    Ray prefixes each line with the process tag and other processes can write
    in between. Lines from the printing process that continue the printed dict
    are kept. Returns ``None`` when no complete configuration is found. In that
    case save the configuration yourself, e.g. ``OmegaConf.save(config, path)``,
    and load it with :func:`pth.config.load_config`.
    """
    buf, owner = [], None
    for line in _lines(path):
        m = _RAY_PREFIX.match(line)
        tag, body = (m.group(1), m.group(2)) if m else ("", line)
        if owner is None:
            if body.startswith("{'actor_rollout_ref'") or body.startswith("{'algorithm'"):
                owner = tag
                buf.append(body)
                if body.rstrip().endswith("}"):
                    try:
                        return ast.literal_eval(body)
                    except (SyntaxError, ValueError):
                        pass
            continue
        if tag != owner or not _CONTINUATION.match(body):
            continue
        buf.append(body)
        if body.rstrip().endswith("}"):
            try:
                cfg = ast.literal_eval("\n".join(buf))
            except (SyntaxError, ValueError):
                continue
            return cfg if isinstance(cfg, dict) else None
    return None


def _get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def static_check(cfg: Mapping[str, Any]) -> Report:
    """Check 1 from the configuration alone: where ``pi_old`` comes from in verl.

    With ``bypass_mode`` off, verl recomputes ``pi_old`` with the learner at the
    start of the update. With one epoch and one mini-batch, the single optimiser
    step is then taken at the parameters that computed ``pi_old``, the PPO ratio
    is 1 on every token and the clip cannot act on sampler lag. That is the
    standard on-policy setup and correct there. Under any sampler lag (delayed
    weight sync, asynchronous rollout, replay) the correction then has to come
    from ``algorithm.rollout_correction.rollout_is``.
    """
    rep = Report(subject="check 1 (config): source of pi_old")
    epochs = _get(cfg, "actor_rollout_ref.actor.ppo_epochs")
    mini = _get(cfg, "actor_rollout_ref.actor.ppo_mini_batch_size")
    batch = _get(cfg, "data.train_batch_size")
    rc = _get(cfg, "algorithm.rollout_correction") or {}
    bypass = bool(rc.get("bypass_mode")) or bool(_get(cfg, "actor_rollout_ref.actor.use_rollout_log_probs"))
    rollout_is = rc.get("rollout_is")
    threshold = rc.get("rollout_is_threshold")
    recorded = _get(cfg, "actor_rollout_ref.rollout.calculate_log_probs")
    rep.stats.update(ppo_epochs=epochs, ppo_mini_batch_size=mini, train_batch_size=batch,
                     bypass_mode=bypass, rollout_is=rollout_is, calculate_log_probs=recorded)

    if None in (epochs, mini, batch):
        rep.add(1, "warn", "configuration lacks ppo_epochs, ppo_mini_batch_size or train_batch_size")
        return rep

    single_step = epochs == 1 and mini >= batch
    rep.stats["ratio_is_one"] = single_step and not bypass
    if not recorded:
        rep.add(1, "warn", "the sampler's log-probabilities are not recorded",
                "Set actor_rollout_ref.rollout.calculate_log_probs=True so the ratio against the sampler "
                "can be logged and corrected.")
    if bypass:
        rep.add(1, "info", "pi_old is the sampler's recorded policy (bypass mode)")
    elif single_step:
        corr = (f"truncated importance weights, rollout_is={rollout_is}, threshold {threshold}"
                if rollout_is else "none")
        rep.add(1, "warn" if not rollout_is else "info",
                "the PPO ratio is 1 by construction: pi_old is recomputed and each batch takes one step",
                f"ppo_epochs={epochs}, ppo_mini_batch_size={mini}, train_batch_size={batch}, bypass_mode off. "
                f"Correction for sampler lag: {corr}."
                + ("" if rollout_is else " Under sampler lag this arm trains without off-policy correction."))
    else:
        rep.add(1, "info", f"pi_old is recomputed by the learner, {epochs} epoch(s) over "
                f"{-(-batch // mini)} mini-batch(es)",
                "The clip acts on the learner's drift within an update, not on sampler lag.")
    return rep


def check_log(path: str, kl_tol: float = 0.05) -> Report:
    """Run check 1 on one verl console log: configuration and logged series."""
    rep = Report(subject=f"{path}")
    cfg = read_config(path)
    metrics = read_metrics(path)
    rep.stats["updates_logged"] = len(metrics)
    if cfg is None:
        rep.add(1, "warn", "no resolved configuration found in the log",
                "Static checks skipped. Save the configuration at launch to enable them.")
    else:
        rep.extend(static_check(cfg))

    clip = {s: r[CLIP_KEY] for s, r in metrics.items() if CLIP_KEY in r}
    kl = {s: r[KL_KEY] for s, r in metrics.items() if KL_KEY in r}
    weighted = any(IS_KEY in r for r in metrics.values())
    if cfg is not None:
        weighted = weighted or bool(_get(cfg, "algorithm.rollout_correction.rollout_is"))
    if not metrics:
        rep.add(1, "warn", "no step lines found in the log")
    elif not kl:
        rep.add(1, "warn", f"{KL_KEY} is not logged",
                "Record the sampler's log-probabilities (rollout.calculate_log_probs=True) to measure the lag.")
    else:
        rep.extend(scan_clip_vs_kl(clip, kl, kl_tol=kl_tol, is_weighted=weighted))
    return rep
