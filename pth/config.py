"""PTH check 2: check the resolved configuration per arm.

Run labels say what each run was meant to be. The resolved configuration says
what it was. A comparison holds when the fields it varies differ between arms,
the fields it holds fixed match, and every arm receives the same set of data
seeds. The ordered prompt IDs of the first batches confirm the data order
directly.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

from .report import Report

DEFAULT_IGNORE = (
    "trainer.experiment_name",
    "trainer.project_name",
    "trainer.default_local_dir",
    "trainer.rollout_data_dir",
    "trainer.validation_data_dir",
    "*.output_dir",
    "*.run_name",
    "*.wandb*",
)

_MISSING = "<missing>"


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings into dotted keys. Lists stay as values."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a resolved configuration from JSON or YAML (needs PyYAML).

    For verl console logs use :func:`pth.verl.read_config`.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("loading YAML configs needs PyYAML: pip install pyyaml") from e
    return yaml.safe_load(text)


def _ignored(key: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, p) for p in patterns)


def _fmt(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


def check_arms(
    runs: Mapping[str, Mapping[str, Any]],
    arm_of: Union[Mapping[str, str], Callable[[str], str]],
    vary: Sequence[str] = (),
    replicate: Sequence[str] = ("data.seed",),
    ignore: Sequence[str] = DEFAULT_IGNORE,
) -> Report:
    """Compare the resolved configurations of every run in a comparison.

    Args:
        runs: run name to its resolved configuration (nested or already flat).
        arm_of: run name to arm name, as a mapping or a function.
        vary: fields the comparison is meant to vary between arms.
        replicate: fields that distinguish replicates within an arm, such as
            the data seed. Every arm should use the same set of values.
        ignore: glob patterns for fields that differ by design (names, paths).
    """
    rep = Report(subject="check 2: configuration of each arm")
    if not runs:
        rep.add(2, "warn", "no runs given")
        return rep
    get_arm = arm_of.get if isinstance(arm_of, Mapping) else arm_of
    flat = {name: flatten(cfg) for name, cfg in runs.items()}
    arms: Dict[str, list] = defaultdict(list)
    for name in flat:
        arm = get_arm(name)
        if arm is None:
            raise KeyError(f"no arm given for run {name!r}")
        arms[arm].append(name)
    rep.stats["arms"] = {a: sorted(ns) for a, ns in arms.items()}

    all_keys = sorted(set().union(*flat.values()))
    skip = set(vary) | set(replicate)

    # Replicates: distinct within each arm, same set across arms.
    for key in replicate:
        per_arm = {}
        for arm, names in sorted(arms.items()):
            values = [flat[n].get(key, _MISSING) for n in sorted(names)]
            per_arm[arm] = values
            if len(names) > 1 and len(set(map(repr, values))) < len(values):
                counts = defaultdict(list)
                for n, v in zip(sorted(names), values):
                    counts[repr(v)].append(n)
                shared = {v: ns for v, ns in counts.items() if len(ns) > 1}
                desc = "; ".join(f"{key}={v} in {', '.join(ns)}" for v, ns in shared.items())
                rep.add(
                    2, "fail",
                    f"arm '{arm}': {sum(len(ns) for ns in shared.values())} runs share {key}",
                    f"{desc}. These runs repeat one data order, so their spread measures sampling noise only.",
                    arm=arm, key=key,
                )
        value_sets = {arm: frozenset(map(repr, vs)) for arm, vs in per_arm.items()}
        if len(set(value_sets.values())) > 1:
            desc = "; ".join(f"{arm}: {sorted(vs)}" for arm, vs in sorted(value_sets.items()))
            rep.add(
                2, "fail",
                f"arms use different sets of {key}",
                f"{desc}. Results cannot be paired by {key} across these arms.",
                key=key,
            )
        rep.stats.setdefault("replicates", {})[key] = {a: [repr(v) for v in vs] for a, vs in per_arm.items()}

    # Declared variations must actually vary.
    for key in vary:
        reps = {arm: sorted({repr(flat[n].get(key, _MISSING)) for n in names}) for arm, names in arms.items()}
        if len(arms) > 1 and len({tuple(v) for v in reps.values()}) == 1:
            rep.add(2, "fail", f"{key} is meant to vary but is identical across arms",
                    f"Every arm has {reps[next(iter(reps))]}. The setting did not reach the runs.", key=key)
        else:
            rep.add(2, "info", f"{key} varies as declared",
                    "; ".join(f"{a}: {', '.join(v)}" for a, v in sorted(reps.items())))

    # Fields that differ without being declared.
    between, within = [], []
    for key in all_keys:
        if key in skip or _ignored(key, ignore):
            continue
        arm_values = {}
        for arm, names in arms.items():
            vals = {repr(flat[n].get(key, _MISSING)) for n in names}
            if len(vals) > 1:
                within.append((key, arm, sorted(vals)))
            arm_values[arm] = sorted(vals)
        if len({tuple(v) for v in arm_values.values()}) > 1:
            between.append((key, arm_values))

    for key, arm_values in between:
        rep.add(2, "warn", f"{key} differs between arms without being declared",
                "; ".join(f"{a}: {', '.join(v)}" for a, v in sorted(arm_values.items())), key=key)
    for key, arm, vals in within:
        rep.add(2, "warn", f"{key} differs within arm '{arm}'", ", ".join(vals), key=key, arm=arm)

    if rep.ok and not between and not within:
        rep.add(2, "info", f"{len(flat)} runs in {len(arms)} arms: only declared fields differ")
    return rep


def data_order_fingerprint(prompt_ids: Iterable[Any]) -> str:
    """Hash the ordered prompt IDs of a run's first batches.

    Pass any stable identifier per prompt: a dataset index, a UID, or the
    prompt text itself.
    """
    h = hashlib.sha1()
    for pid in prompt_ids:
        h.update(repr(pid).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def check_data_orders(
    fingerprints: Mapping[str, str],
    seeds: Optional[Mapping[str, Any]] = None,
) -> Report:
    """Confirm that runs labelled with different seeds see different data orders.

    Args:
        fingerprints: run name to :func:`data_order_fingerprint` of its first batches.
        seeds: run name to the data seed it was launched with.
    """
    rep = Report(subject="check 2: data order of each run")
    groups: Dict[str, list] = defaultdict(list)
    for name, fp in fingerprints.items():
        groups[fp].append(name)
    rep.stats["distinct_orders"] = len(groups)
    rep.stats["runs"] = len(fingerprints)
    seeds = seeds or {}

    for fp, names in groups.items():
        if len(names) < 2:
            continue
        labelled = {repr(seeds.get(n, _MISSING)) for n in names}
        if len(labelled) > 1 or not seeds:
            rep.add(
                2, "fail" if seeds else "warn",
                f"{len(names)} runs share one data order: {', '.join(sorted(names))}",
                ("Their launch seeds differ (" + ", ".join(f"{n}={_fmt(seeds.get(n, _MISSING))}" for n in sorted(names))
                 + "), so the seed did not reach at least one of them.") if seeds else
                "If these runs are meant as different seeds, the seed did not reach them.",
                fingerprint=fp,
            )
    by_seed: Dict[str, set] = defaultdict(set)
    for name, fp in fingerprints.items():
        if name in seeds:
            by_seed[repr(seeds[name])].add(fp)
    for s, fps in by_seed.items():
        if len(fps) > 1:
            rep.add(2, "warn", f"seed {s} produced {len(fps)} different data orders",
                    "The data order depends on something besides the seed.")
    if rep.ok and len(groups) == len(fingerprints):
        rep.add(2, "info", f"{len(fingerprints)} runs, {len(groups)} distinct data orders")
    return rep
