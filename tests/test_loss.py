import os
import runpy

import pytest

torch = pytest.importorskip("torch")

from pth import check_loss, random_batch  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "examples")


def test_random_batch_is_reproducible():
    a, b = random_batch(seed=3), random_batch(seed=3)
    for k in a:
        assert torch.equal(a[k], b[k])
    assert (a["mask"].sum(1) >= 1).all()


def test_normaliser_mismatch_is_diagnosed():
    ns = runpy.run_path(os.path.join(EXAMPLES, "loss_vs_equation.py"))
    rep = check_loss(ns["tis_implemented"], ns["tis_written"], random_batch(), name="tis")
    assert not rep.ok
    grad = [f for f in rep.by_level("fail") if "gradient" in f.title]
    assert grad and "normaliser" in grad[0].detail
    assert grad[0].tag == "Rogue Normaliser"


def test_matching_loss_passes():
    ns = runpy.run_path(os.path.join(EXAMPLES, "loss_vs_equation.py"))
    assert check_loss(ns["tis_fixed"], ns["tis_written"], random_batch()).ok


def test_global_scale_is_diagnosed():
    def eq(logp, advantages, mask, **_):
        return -(advantages[:, None] * logp * mask).sum()

    def impl(logp, advantages, mask, **_):
        return -0.5 * (advantages[:, None] * logp * mask).sum()

    rep = check_loss(impl, eq, random_batch())
    assert "global factor 0.5" in rep.by_level("fail")[-1].detail
    assert rep.by_level("fail")[-1].tag == ""


def test_wrong_direction():
    def eq(logp, advantages, mask, **_):
        return -(advantages[:, None] * logp * mask).sum()

    def impl(logp, advantages, mask, **_):
        return -(advantages[:, None] * logp.exp() * mask).sum()

    rep = check_loss(impl, eq, random_batch())
    assert "Cosine similarity" in rep.by_level("fail")[-1].detail
