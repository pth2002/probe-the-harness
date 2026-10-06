"""PTH check 1: log the ratio against the policy that generated the data.

The PPO ratio corrects for sampler lag only when its denominator is the
probability the sampler recorded. A ratio taken against probabilities the
learner recomputed at the start of the update stays at 1 when each batch takes
one optimiser step, so the clip never acts however far the sampler lags.
The clip fraction alone cannot tell these two setups apart. The ratio against
the sampler can.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ._arrays import masked
from .report import Report

_QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)


def _ratio_summary(log_ratio: np.ndarray, clip_eps: float) -> dict:
    r = np.exp(np.clip(log_ratio, -20.0, 20.0))
    out = {f"p{int(q * 100):02d}": float(np.quantile(r, q)) for q in _QUANTILES}
    out["min"] = float(r.min())
    out["max"] = float(r.max())
    out["outside_clip_range"] = float(np.mean((r < 1.0 - clip_eps) | (r > 1.0 + clip_eps)))
    out["max_abs_log_ratio"] = float(np.abs(log_ratio).max())
    return out


def ratio_report(
    logp_learner: Any,
    logp_sampler: Any,
    mask: Optional[Any] = None,
    logp_old: Optional[Any] = None,
    clip_eps: float = 0.2,
    kl_tol: float = 0.05,
    one_tol: float = 1e-6,
) -> Report:
    """Measure the ratio a PPO-style loss uses against the ratio to the sampler.

    Args:
        logp_learner: log-probabilities of the sampled tokens under the policy
            being updated, shape ``(batch, length)``.
        logp_sampler: log-probabilities the sampler recorded for the same tokens.
        mask: 1 for response tokens, 0 for padding.
        logp_old: the ``pi_old`` the loss puts in its ratio, when it is not the
            sampler's probability (for example verl's ``old_log_probs``).
        clip_eps: PPO clip range.
        kl_tol: KL(sampler || pi_old), in nats per token, above which the
            sampler counts as lagging. Estimated with the nonnegative k3
            estimator over the sampled tokens.
        one_tol: largest ``|log pi_theta - log pi_old|`` treated as a ratio of 1.

    Returns:
        A :class:`Report` whose ``stats`` hold the ratio quantiles and the KL.
    """
    lp = masked(logp_learner, mask)
    lq = masked(logp_sampler, mask)
    if lp.size == 0:
        raise ValueError("no response tokens under the mask")

    rep = Report(subject="check 1: policy in the PPO ratio")
    ref = masked(logp_old, mask) if logp_old is not None else lp
    log_r = np.clip(ref - lq, -20.0, 20.0)
    # k3 estimator of KL(sampler || pi_old) on tokens drawn from the sampler: nonnegative per token.
    kl = float(np.mean(np.expm1(log_r) - log_r))
    rep.stats["tokens"] = int(lp.size)
    rep.stats["kl_sampler_to_old"] = kl
    rep.stats["mean_log_q_minus_log_old"] = float(np.mean(lq - ref))
    rep.stats["ratio_to_sampler"] = _ratio_summary(lp - lq, clip_eps)

    lagging = kl > kl_tol
    to_sampler = rep.stats["ratio_to_sampler"]

    if logp_old is None:
        level = "warn" if lagging else "info"
        rep.add(
            1, level,
            f"sampler-learner KL {kl:.3g} nat/token, {to_sampler['outside_clip_range']:.1%} of tokens "
            f"outside [1-{clip_eps}, 1+{clip_eps}] against the sampler",
            "Pass logp_old to compare with the ratio the loss actually uses." if lagging else "",
        )
        return rep

    lo = masked(logp_old, mask)
    to_old = _ratio_summary(lp - lo, clip_eps)
    rep.stats["ratio_to_old"] = to_old
    ratio_is_one = to_old["max_abs_log_ratio"] <= one_tol

    if ratio_is_one and lagging:
        rep.add(
            1, "fail",
            "the PPO ratio is 1 on every token while the sampler lags",
            f"KL(sampler || pi_old) = {kl:.3g} nat/token and max |log pi_theta - log pi_old| = "
            f"{to_old['max_abs_log_ratio']:.2g}. pi_old is the learner's own recomputed policy, so the clip "
            f"cannot act. Against the sampler, {to_sampler['outside_clip_range']:.1%} of tokens fall outside "
            "the clip range. Take pi_old from the sampler's recorded probabilities or add an importance weight "
            "against them.",
            kl=kl,
        )
    elif lagging and to_old["outside_clip_range"] < 0.1 * max(to_sampler["outside_clip_range"], 1e-12):
        rep.add(
            1, "warn",
            "the clip sees far less lag than the sampler shows",
            f"{to_old['outside_clip_range']:.2%} of tokens outside the clip range against pi_old, "
            f"{to_sampler['outside_clip_range']:.2%} against the sampler. Check where pi_old comes from.",
        )
    else:
        rep.add(
            1, "info",
            f"KL(sampler || pi_old) {kl:.3g} nat/token, clip range exceeded on "
            f"{to_old['outside_clip_range']:.1%} of tokens against pi_old and "
            f"{to_sampler['outside_clip_range']:.1%} against the sampler",
        )
    return rep


def scan_clip_vs_kl(
    clip_fraction: Mapping[int, float],
    kl: Mapping[int, float],
    kl_tol: float = 0.05,
    is_weighted: bool = False,
) -> Report:
    """Check 1 from logged series: a clip fraction of exactly zero next to a lagging sampler.

    Args:
        clip_fraction: update index to logged PPO clip fraction.
        kl: update index to logged sampler-learner KL (nats per token), for
            example verl's ``rollout_corr/kl``.
        kl_tol: KL above which the sampler counts as lagging.
        is_weighted: the loss multiplies in an importance weight against the
            sampler's probabilities (truncated IS). That weight then carries
            the correction, and an inactive clip is expected.
    """
    rep = Report(subject="check 1: clip fraction against sampler lag")
    steps = sorted(set(clip_fraction) & set(kl))
    if not steps:
        rep.add(1, "warn", "no update has both a clip fraction and a sampler-learner KL",
                "Record the sampler's log-probabilities so the lag can be measured.")
        return rep

    clips = np.array([clip_fraction[s] for s in steps], dtype=float)
    kls = np.array([kl[s] for s in steps], dtype=float)
    lag_steps = [s for s, v in zip(steps, kls) if v > kl_tol]
    pos = kls[kls > 0]
    growth = float(pos.max() / pos.min()) if pos.size else float("nan")
    rep.stats.update(
        updates=len(steps),
        max_clip_fraction=float(clips.max()),
        max_kl=float(kls.max()),
        kl_growth=growth,
        lagging_updates=len(lag_steps),
    )

    if not lag_steps:
        rep.add(1, "info", f"sampler and learner stay close (max KL {kls.max():.3g} nat/token)",
                "Under this lag the clip has nothing to correct, so its fraction says little about the setup.")
    elif clips.max() == 0.0 and is_weighted:
        rep.add(1, "info",
                f"clip inactive on all {len(steps)} updates, sampler lag corrected by importance weights",
                f"Sampler-learner KL reaches {kls.max():.3g} nat/token. The importance weight against the "
                "sampler is the only correction in this arm.")
    elif clips.max() == 0.0:
        rep.add(
            1, "fail",
            f"clip fraction is exactly 0 on all {len(steps)} updates while the sampler lags",
            f"Sampler-learner KL reaches {kls.max():.3g} nat/token ({growth:.0f}x its smallest value) and exceeds "
            f"{kl_tol} on {len(lag_steps)} updates. A clip that never acts under this lag points to a ratio "
            "taken against a recomputed policy, which leaves the arm without off-policy correction unless the "
            "loss also applies an importance weight against the sampler. The source of pi_old in the code "
            "settles it.",
            first_lagging_update=lag_steps[0],
        )
    else:
        active = float(np.mean(clips > 0))
        rep.add(1, "info", f"clip acts on {active:.0%} of updates, max KL {kls.max():.3g} nat/token")
    return rep
