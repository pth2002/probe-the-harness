"""PTH check 3: log the identity of the training batch alongside its age.

A replay queue can give every update the intended data age and still train on
the same batch many times, for example while it fills up. The age log looks
right in that case. The count of distinct batches does not.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .report import STUCK_BATCH, Report


def batch_fingerprint(*parts: Any) -> str:
    """Hash the content that identifies a batch (token IDs, prompt IDs, UIDs)."""
    h = hashlib.sha1()
    for part in parts:
        if hasattr(part, "detach"):
            part = part.detach().cpu().numpy()
        if isinstance(part, np.ndarray):
            h.update(str(part.dtype).encode())
            h.update(str(part.shape).encode())
            h.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, (bytes, bytearray)):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class BatchLedger:
    """Record which batch each update trains on.

    Example::

        ledger = BatchLedger()
        for update in range(num_updates):
            batch = queue.next()
            ledger.record(update, batch_fingerprint(batch["input_ids"]), age=batch_age)
            ...
        print(ledger.report())
    """

    def __init__(self, expect_unique: bool = True) -> None:
        self.expect_unique = expect_unique
        self.rows: List[Dict[str, Any]] = []

    def record(self, update: int, batch: Any, age: Optional[int] = None) -> str:
        """Record one update. ``batch`` is a fingerprint string or the batch content."""
        fp = batch if isinstance(batch, str) else batch_fingerprint(batch)
        self.rows.append({"update": int(update), "batch": fp, "age": age})
        return fp

    def report(self, expected_age: Optional[Callable[[int], int]] = None) -> Report:
        return check_batch_ids(
            [r["batch"] for r in self.rows],
            updates=[r["update"] for r in self.rows],
            ages=[r["age"] for r in self.rows] if any(r["age"] is not None for r in self.rows) else None,
            expect_unique=self.expect_unique,
            expected_age=expected_age,
        )


def _runs(ids: Sequence[str]) -> List[tuple]:
    """Maximal stretches of consecutive updates on the same batch: (start index, length)."""
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            out.append((start, i - start))
            start = i
    return out


def check_batch_ids(
    batch_ids: Sequence[Any],
    updates: Optional[Sequence[int]] = None,
    ages: Optional[Sequence[Optional[int]]] = None,
    expect_unique: bool = True,
    expected_age: Optional[Callable[[int], int]] = None,
) -> Report:
    """Count distinct training batches and locate repeats.

    Args:
        batch_ids: identifier of the batch trained on at each update, in order.
        updates: update index of each entry (defaults to 0, 1, 2, ...).
        ages: data age of each entry, if logged.
        expect_unique: the design gives every update a new batch.
        expected_age: the designed data age as a function of the update index.
    """
    rep = Report(subject="check 3: data for each update")
    ids = [str(b) for b in batch_ids]
    n = len(ids)
    if n == 0:
        rep.add(3, "warn", "no updates recorded")
        return rep
    ups = list(updates) if updates is not None else list(range(n))
    distinct = len(set(ids))
    longest_start, longest_len = max(_runs(ids), key=lambda t: t[1])
    rep.stats.update(updates=n, distinct_batches=distinct, longest_repeat=longest_len)

    uses: Dict[str, List[int]] = {}
    for u, b in zip(ups, ids):
        uses.setdefault(b, []).append(u)
    repeated = {b: us for b, us in uses.items() if len(us) > 1}
    rep.stats["repeated_batches"] = len(repeated)

    if expect_unique and distinct < n:
        detail = [f"{n} updates trained on {distinct} distinct batches."]
        if longest_len > 1:
            detail.append(
                f"Updates {ups[longest_start]} to {ups[longest_start + longest_len - 1]} all trained on one batch "
                f"({longest_len} updates in a row)."
            )
        worst = max(repeated.items(), key=lambda kv: len(kv[1]))
        if len(worst[1]) != longest_len:
            detail.append(f"The most reused batch appears at {len(worst[1])} updates.")
        if ages is not None:
            detail.append("The logged data age does not reveal this, since a repeated batch can carry any age.")
        verb = "reuses" if n - distinct == 1 else "reuse"
        rep.add(3, "fail", f"{n - distinct} of {n} updates {verb} an earlier batch", " ".join(detail),
                tag=STUCK_BATCH, first_repeat_update=min(us[1] for us in repeated.values()))
    else:
        rep.add(3, "info", f"{n} updates, {distinct} distinct batches")

    if ages is not None:
        known = [(u, a) for u, a in zip(ups, ages) if a is not None]
        if known:
            rep.stats["age_min"] = min(a for _, a in known)
            rep.stats["age_max"] = max(a for _, a in known)
        if expected_age is not None:
            off = [(u, a, expected_age(u)) for u, a in known if a != expected_age(u)]
            if off:
                u, a, e = off[0]
                rep.add(3, "fail", f"data age departs from the design on {len(off)} updates",
                        f"First at update {u}: age {a}, designed {e}.")
            else:
                rep.add(3, "info", "data age follows the design on every update")
    return rep
