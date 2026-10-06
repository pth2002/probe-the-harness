import os
import runpy

import numpy as np

from pth import BatchLedger, batch_fingerprint, check_batch_ids

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "examples")


def test_warmup_queue_reproduces_report_numbers():
    ns = runpy.run_path(os.path.join(EXAMPLES, "replay_queue.py"))
    ledger = BatchLedger()
    for update, batch_id, age in ns["warmup_queue"](32, 100):
        ledger.record(update, batch_id, age=age)
    rep = ledger.report(expected_age=lambda u: min(u, 32))
    assert rep.stats["distinct_batches"] == 68
    assert rep.stats["longest_repeat"] == 33
    assert [f.level for f in rep.findings] == ["fail", "info"]  # identity fails, age passes
    assert rep.findings[0].tag == "Stuck Batch"


def test_unique_prefill_passes_with_same_ages():
    ns = runpy.run_path(os.path.join(EXAMPLES, "replay_queue.py"))
    ledger = BatchLedger()
    for update, batch_id, age in ns["unique_prefill_queue"](32, 100):
        ledger.record(update, batch_id, age=age)
    rep = ledger.report(expected_age=lambda u: min(u, 32))
    assert rep.ok and rep.stats["distinct_batches"] == 100


def test_fingerprint_depends_on_content():
    x = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert batch_fingerprint(x) == batch_fingerprint(x.copy())
    assert batch_fingerprint(x) != batch_fingerprint(x[::-1])
    assert batch_fingerprint(x) != batch_fingerprint(x.astype(np.int32))


def test_allowed_repeats():
    rep = check_batch_ids(["a", "a", "b", "b"], expect_unique=False)
    assert rep.ok
    assert not check_batch_ids(["a", "a", "b", "b"]).ok
