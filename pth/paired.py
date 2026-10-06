"""PTH check 6: report paired differences per seed.

A pooled statement such as "all nine runs of A finish above all six runs of B"
mixes seed effects with method effects. Pairing runs by data seed separates
them, and shows whether the arms ran the same seeds at all.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any, Dict, Mapping

import numpy as np

from .report import Report


def paired_differences(
    results: Mapping[str, Mapping[Any, float]],
    a: str,
    b: str,
    scale: float = 1.0,
) -> Report:
    """Per-seed differences ``a - b``.

    Args:
        results: method name to ``{seed: final metric}``.
        a, b: the two methods to compare.
        scale: multiply differences by this for display, e.g. 100 for points.
    """
    rep = Report(subject=f"check 6: {a} minus {b}, paired by seed")
    if a not in results or b not in results:
        missing = [m for m in (a, b) if m not in results]
        raise KeyError(f"no results for {missing}")
    ra, rb = results[a], results[b]
    common = sorted(set(ra) & set(rb), key=repr)
    only_a = sorted(set(ra) - set(rb), key=repr)
    only_b = sorted(set(rb) - set(ra), key=repr)

    if not common:
        rep.add(6, "fail", f"{a} and {b} share no seed",
                "The arms cannot be paired. Rerun them on a common set of seeds.")
        return rep

    diffs = {s: (float(ra[s]) - float(rb[s])) * scale for s in common}
    d = np.array(list(diffs.values()))
    rep.stats.update(
        pairs=len(common),
        differences={repr(s): v for s, v in diffs.items()},
        mean=float(d.mean()),
        min=float(d.min()),
        max=float(d.max()),
        a_ahead=int((d > 0).sum()),
        b_ahead=int((d < 0).sum()),
    )
    pooled_a_all_above = min(map(float, ra.values())) > max(map(float, rb.values()))
    pooled_b_all_above = min(map(float, rb.values())) > max(map(float, ra.values()))
    rep.stats["pooled_separation"] = a if pooled_a_all_above else (b if pooled_b_all_above else None)

    pairs = ", ".join(f"{s}: {v:+.3g}" for s, v in diffs.items())
    rep.add(6, "info", f"mean {d.mean():+.3g} over {len(common)} seeds, range {d.min():+.3g} to {d.max():+.3g}",
            f"{pairs}. {a} ahead on {(d > 0).sum()}, {b} ahead on {(d < 0).sum()}.")
    if only_a or only_b:
        rep.add(6, "warn", "some seeds ran in one arm only",
                f"only {a}: {only_a or 'none'}. Only {b}: {only_b or 'none'}. Unpaired runs are left out above, "
                "and a pooled ordering that includes them mixes seed effects with method effects.")
    return rep


def read_results_csv(path: str, method: str = "method", seed: str = "seed", value: str = "value") -> Dict[str, Dict[str, float]]:
    """Read ``method,seed,value`` rows into the mapping :func:`paired_differences` takes."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row[method]][row[seed]] = float(row[value])
    return dict(out)
