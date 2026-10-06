import numpy as np

from pth import ratio_report, scan_clip_vs_kl


def _batch(seed=0, lag=1.0):
    rng = np.random.default_rng(seed)
    logp = -rng.uniform(0.01, 4.0, size=(8, 20))
    logq = logp - lag * np.abs(rng.normal(size=logp.shape))
    mask = np.ones_like(logp)
    mask[:, 15:] = 0
    return logp, logq, mask


def test_ratio_against_recomputed_policy_fails_under_lag():
    logp, logq, mask = _batch()
    rep = ratio_report(logp, logq, mask, logp_old=logp.copy())
    assert not rep.ok
    assert rep.stats["ratio_to_old"]["max_abs_log_ratio"] == 0.0
    assert rep.stats["ratio_to_sampler"]["outside_clip_range"] > 0.5


def test_ratio_against_sampler_passes():
    logp, logq, mask = _batch()
    rep = ratio_report(logp, logq, mask, logp_old=logq)
    assert rep.ok
    assert rep.stats["ratio_to_old"]["outside_clip_range"] == rep.stats["ratio_to_sampler"]["outside_clip_range"]


def test_no_lag_is_informational():
    logp, _, mask = _batch()
    rep = ratio_report(logp, logp + 1e-4, mask, logp_old=logp)
    assert rep.ok
    assert all(f.level == "info" for f in rep.findings)


def test_mask_excludes_padding():
    logp, logq, mask = _batch()
    logq[:, 15:] = -1e9  # garbage in padding must not matter
    rep = ratio_report(logp, logq, mask)
    assert np.isfinite(rep.stats["kl_sampler_to_old"])
    assert rep.stats["tokens"] == 8 * 15


def test_scan_flags_zero_clip_under_lag():
    steps = range(100)
    clip = {s: 0.0 for s in steps}
    kl = {s: 1e-3 * (1.1 ** s) for s in steps}
    rep = scan_clip_vs_kl(clip, kl)
    assert not rep.ok
    assert rep.stats["max_clip_fraction"] == 0.0


def test_scan_accepts_importance_weighted_arm():
    steps = range(100)
    rep = scan_clip_vs_kl({s: 0.0 for s in steps}, {s: 0.5 for s in steps}, is_weighted=True)
    assert rep.ok


def test_scan_accepts_active_clip_and_small_lag():
    steps = range(50)
    assert scan_clip_vs_kl({s: 0.01 for s in steps}, {s: 0.5 for s in steps}).ok
    assert scan_clip_vs_kl({s: 0.0 for s in steps}, {s: 1e-3 for s in steps}).ok
