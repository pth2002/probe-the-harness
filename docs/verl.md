# Using PTH with verl

## From console logs

Launch with the console logger and with the sampler's log-probabilities recorded:

```bash
python -m verl.trainer.main_ppo ... \
    trainer.logger='[console]' \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    2>&1 | tee logs/my-run.txt
```

Then run checks 1 and 2 on the saved logs:

```bash
pth verl logs/*.txt                                    # check 1 per run
pth verl --arm "a=logs/a-*.txt" --arm "b=logs/b-*.txt" \
         --vary <field the comparison varies>          # adds check 2 across arms
```

Check 1 reads `actor/pg_clipfrac` and `rollout_corr/kl` from every `step:N` line and the PPO settings from the printed configuration. An arm counts as importance-weighted when its log contains `rollout_corr/rollout_is_mean` or its configuration sets `algorithm.rollout_correction.rollout_is`. Adjust the lag threshold with `--kl-tol` (default 0.05 nat per token).

## Check 1 inside the update

`verl/workers/actor/dp_actor.py`, in `update_policy`, has `log_prob` (current policy), `old_log_prob` (the pi_old the loss uses) and `response_mask` for every micro-batch, and the sampler's log-probabilities in `model_inputs["rollout_log_probs"]` when `calculate_log_probs=True`. After `old_log_prob` is set:

```python
from pth import ratio_report

rollout_log_prob = model_inputs.get("rollout_log_probs")
if rollout_log_prob is not None:
    rep = ratio_report(log_prob, rollout_log_prob, response_mask, logp_old=old_log_prob)
    micro_batch_metrics["pth/kl_sampler_to_old"] = rep.stats["kl_sampler_to_old"]
    micro_batch_metrics["pth/ratio_to_sampler_p99"] = rep.stats["ratio_to_sampler"]["p99"]
    micro_batch_metrics["pth/ratio_to_sampler_outside_clip"] = rep.stats["ratio_to_sampler"]["outside_clip_range"]
    micro_batch_metrics["pth/ratio_to_old_max_abs_log"] = rep.stats["ratio_to_old"]["max_abs_log_ratio"]
```

`pth/ratio_to_old_max_abs_log` at zero next to a growing `pth/kl_sampler_to_old` is the signature of the Idle Clip. File and variable names follow verl releases from late 2025 and may move between versions.

## Configurations and data orders for check 2

verl prints the resolved configuration at start-up, and `pth verl` recovers it from the log. To keep a clean copy, save it at launch:

```python
from omegaconf import OmegaConf
OmegaConf.save(config, "resolved_config.yaml", resolve=True)
```

and load it with `pth.load_config("resolved_config.yaml")`.

For the data order, log a stable identifier for each prompt of the first few batches, such as the dataset index your preprocessing stores in `extra_info` or the prompt text, and compare runs with `pth.data_order_fingerprint` and `pth.check_data_orders`. Runs launched with different seeds should produce different fingerprints.
