import copy

from pth import check_arms, check_data_orders, data_order_fingerprint, flatten

BASE = {
    "data": {"seed": None, "train_batch_size": 16},
    "algorithm": {"rollout_correction": {"rollout_is": None, "rollout_is_threshold": 2.0}},
    "actor_rollout_ref": {"actor": {"optim": {"lr": 2e-6}}},
    "trainer": {"experiment_name": "x"},
}


def run(seed, rollout_is=None, lr=2e-6, name="x"):
    cfg = copy.deepcopy(BASE)
    cfg["data"]["seed"] = seed
    cfg["algorithm"]["rollout_correction"]["rollout_is"] = rollout_is
    cfg["actor_rollout_ref"]["actor"]["optim"]["lr"] = lr
    cfg["trainer"]["experiment_name"] = name
    return cfg


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": [1, 2]}}}) == {"a.b": 1, "a.c.d": [1, 2]}


def test_seed_that_never_reached_one_arm():
    # The launch script overwrote the seed for the importance-sampling arm only.
    runs = {
        "grpo-r1": run(None, name="grpo-r1"), "grpo-r2": run(43, name="grpo-r2"), "grpo-r3": run(44, name="grpo-r3"),
        "tis-r1": run(None, "token", name="tis-r1"), "tis-r2": run(None, "token", name="tis-r2"),
        "tis-r3": run(None, "token", name="tis-r3"),
    }
    rep = check_arms(runs, lambda n: n.split("-")[0], vary=["algorithm.rollout_correction.rollout_is"])
    titles = [f.title for f in rep.by_level("fail")]
    assert any("arm 'tis': 3 runs share data.seed" in t for t in titles)
    assert any("different sets of data.seed" in t for t in titles)
    assert {f.tag for f in rep.by_level("fail")} == {"Lost Seed"}
    assert not rep.by_level("warn")  # experiment names are ignored by default


def test_clean_comparison_passes():
    runs = {f"{arm}-{s}": run(s, ris) for arm, ris in (("grpo", None), ("tis", "token")) for s in (None, 43, 44)}
    rep = check_arms(runs, lambda n: n.split("-")[0], vary=["algorithm.rollout_correction.rollout_is"])
    assert rep.ok
    assert not rep.by_level("warn")


def test_declared_variation_that_does_not_vary():
    runs = {f"{arm}-{s}": run(s, None) for arm in ("grpo", "tis") for s in (None, 43)}
    rep = check_arms(runs, lambda n: n.split("-")[0], vary=["algorithm.rollout_correction.rollout_is"])
    assert any("meant to vary" in f.title for f in rep.by_level("fail"))


def test_undeclared_difference_is_reported():
    runs = {f"grpo-{s}": run(s) for s in (None, 43)}
    runs.update({f"tis-{s}": run(s, "token", lr=4e-6) for s in (None, 43)})
    rep = check_arms(runs, lambda n: n.split("-")[0], vary=["algorithm.rollout_correction.rollout_is"])
    assert rep.ok
    assert any("optim.lr differs between arms" in f.title for f in rep.by_level("warn"))


def test_data_orders():
    a = data_order_fingerprint([3, 1, 4, 1, 5])
    b = data_order_fingerprint([2, 7, 1, 8, 2])
    assert a != b and a == data_order_fingerprint([3, 1, 4, 1, 5])
    rep = check_data_orders({"r1": a, "r2": a, "r3": b}, seeds={"r1": None, "r2": 43, "r3": 44})
    assert not rep.ok
    assert check_data_orders({"r1": a, "r3": b}, seeds={"r1": None, "r3": 44}).ok
