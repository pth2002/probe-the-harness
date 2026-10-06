"""Probe the Harness (PTH): setup checks for stale-data RL comparisons.

For every layer of the harness, measure the quantity that defines the
comparison alongside the quantities that are convenient to log.

    check 1  ratio_report, scan_clip_vs_kl     policy in the PPO ratio
    check 2  check_arms, check_data_orders     configuration of each arm
    check 3  BatchLedger, check_batch_ids      data for each update
    check 4  check_loss, random_batch          implemented loss (needs torch)
    check 6  paired_differences                paired differences per seed

Checks 5 and 7 are procedures, described in CHECKLIST.md.
"""
from .report import Finding, HarnessCheckError, Report
from .ratio import ratio_report, scan_clip_vs_kl
from .config import check_arms, check_data_orders, data_order_fingerprint, flatten, load_config
from .batches import BatchLedger, batch_fingerprint, check_batch_ids
from .paired import paired_differences, read_results_csv


def __getattr__(name):
    if name in ("check_loss", "random_batch"):
        from . import loss
        return getattr(loss, name)
    raise AttributeError(f"module 'pth' has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "Finding", "HarnessCheckError", "Report",
    "ratio_report", "scan_clip_vs_kl",
    "check_arms", "check_data_orders", "data_order_fingerprint", "flatten", "load_config",
    "BatchLedger", "batch_fingerprint", "check_batch_ids",
    "check_loss", "random_batch",
    "paired_differences", "read_results_csv",
]
