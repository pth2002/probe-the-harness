import pprint

import pytest

from pth import verl
from pth.cli import main

TAG = "\x1b[36m(TaskRunner pid=4242)\x1b[0m "
OTHER = "\x1b[36m(WorkerDict pid=777)\x1b[0m "


def make_config(seed=None, rollout_is=None, epochs=1, mini=16):
    return {
        "actor_rollout_ref": {
            "actor": {"_target_": "verl.workers.config.FSDPActorConfig", "ppo_epochs": epochs,
                      "ppo_mini_batch_size": mini, "optim": {"lr": 2e-06},
                      "policy_loss": {"loss_mode": "vanilla"}, "use_kl_loss": True},
            "rollout": {"calculate_log_probs": True, "n": 8,
                        "trace": {"_target_": "verl.workers.config.TraceConfig", "backend": None}},
        },
        "algorithm": {"adv_estimator": "grpo",
                      "rollout_correction": {"bypass_mode": False, "rollout_is": rollout_is,
                                             "rollout_is_threshold": 2.0}},
        "data": {"seed": seed, "train_batch_size": 16},
        "trainer": {"experiment_name": "run"},
    }


def write_log(path, cfg, lag=True, clip=0.0, weighted=False, steps=40):
    lines = ["ray init kwargs: {'num_cpus': None}", TAG + "TaskRunner hostname: box, PID: 4242"]
    printed = pprint.pformat(cfg, width=60).splitlines()
    for i, line in enumerate(printed):
        lines.append(TAG + line)
        if i == 3:  # another process writes in between
            lines.append(OTHER + "WARNING:2026-01-01 00:00:00,000:Setting TOKENIZERS_PARALLELISM=false")
        if i == 5:  # the same process writes a warning in between
            lines.append(TAG + "/verl/utils/profiler/config.py:49: UserWarning: not supported")
            lines.append(TAG + '  warnings.warn("not supported", stacklevel=1)')
    for s in range(1, steps + 1):
        kl = (1e-3 * 1.3 ** s) if lag else 1e-3
        items = [f"step:{s}", f"actor/pg_clipfrac:{clip}", f"rollout_corr/kl:{kl}", "actor/pg_loss:-0.01"]
        if weighted:
            items.append("rollout_corr/rollout_is_mean:1.01")
        items.append("val-core/openai/gsm8k/reward/mean@1:0.75")
        lines.append(TAG + " - ".join(items))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_read_config_survives_interleaving(tmp_path):
    cfg = make_config(seed=43)
    p = write_log(tmp_path / "a.log", cfg)
    assert verl.read_config(p) == cfg


def test_read_config_wrapped_values(tmp_path):
    # verl's printer can break a nested value onto a line with no indentation.
    cfg = make_config()
    text = pprint.pformat(cfg, width=60).replace("'trace': {", "'trace': \n{")
    p = tmp_path / "w.log"
    p.write_text("\n".join(TAG + l for l in text.splitlines()) + "\n", encoding="utf-8")
    assert verl.read_config(str(p)) == cfg


def test_read_metrics(tmp_path):
    p = write_log(tmp_path / "a.log", make_config(), steps=5)
    m = verl.read_metrics(p)
    assert sorted(m) == [1, 2, 3, 4, 5]
    assert m[3]["val-core/openai/gsm8k/reward/mean@1"] == pytest.approx(0.75)


def test_uncorrected_arm_under_lag_fails(tmp_path):
    rep = verl.check_log(write_log(tmp_path / "grpo.log", make_config()))
    assert not rep.ok
    assert any("ratio is 1 by construction" in f.title for f in rep.by_level("warn"))


def test_importance_weighted_arm_passes(tmp_path):
    rep = verl.check_log(write_log(tmp_path / "tis.log", make_config(rollout_is="token"), weighted=True))
    assert rep.ok


def test_on_policy_run_passes(tmp_path):
    assert verl.check_log(write_log(tmp_path / "fresh.log", make_config(), lag=False)).ok


def test_multi_epoch_config(tmp_path):
    rep = verl.static_check(make_config(epochs=4, mini=4))
    assert rep.stats["ratio_is_one"] is False
    assert rep.ok


def test_cli_end_to_end(tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    for s in (None, 43, 44):
        write_log(logs / f"grpo-{s}.log", make_config(seed=s), lag=False)
        write_log(logs / f"tis-{s}.log", make_config(seed=None, rollout_is="token"), lag=False, weighted=True)
    code = main(["verl", "--arm", f"grpo={logs}/grpo-*.log", "--arm", f"tis={logs}/tis-*.log",
                 "--vary", "algorithm.rollout_correction.rollout_is", "-q"])
    out = capsys.readouterr().out
    assert code == 1
    assert "arm 'tis': 3 runs share data.seed" in out
