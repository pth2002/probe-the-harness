import pytest

from pth import paired_differences


def test_paired_differences():
    results = {"A": {"s1": 0.805, "s2": 0.806, "s3": 0.787}, "B": {"s1": 0.763, "s2": 0.785, "s3": 0.808}}
    rep = paired_differences(results, "A", "B", scale=100)
    assert rep.ok
    assert rep.stats["pairs"] == 3
    assert rep.stats["a_ahead"] == 2 and rep.stats["b_ahead"] == 1
    assert rep.stats["differences"]["'s1'"] == pytest.approx(4.2)
    assert rep.stats["pooled_separation"] is None


def test_pooled_separation_is_recorded():
    results = {"A": {1: 0.84, 2: 0.85, 3: 0.86}, "B": {1: 0.83, 2: 0.82, 3: 0.835}}
    rep = paired_differences(results, "A", "B")
    assert rep.stats["pooled_separation"] == "A"
    assert rep.ok


def test_unpaired_seeds_warn():
    # One arm repeated a single data order, the other ran three.
    results = {"A": {None: 0.805, 43: 0.806, 44: 0.787}, "B": {None: 0.763}}
    rep = paired_differences(results, "A", "B")
    assert rep.stats["pairs"] == 1
    assert any("one arm only" in f.title for f in rep.by_level("warn"))


def test_no_common_seed_fails():
    rep = paired_differences({"A": {1: 0.8}, "B": {2: 0.7}}, "A", "B")
    assert not rep.ok


def test_unknown_method():
    with pytest.raises(KeyError):
        paired_differences({"A": {1: 0.8}}, "A", "B")
