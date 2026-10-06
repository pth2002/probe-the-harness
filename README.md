# Probe the Harness (PTH)

[![arXiv](https://img.shields.io/badge/arXiv-2610.02911-b31b1b.svg)](https://arxiv.org/abs/2610.02911)
[![tests](https://github.com/pth2002/probe-the-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/pth2002/probe-the-harness/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Setup checks for stale-data RL comparisons in language models.

This repository accompanies the technical report *Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models* ([arXiv:2610.02911](https://arxiv.org/abs/2610.02911)).

Every comparison between RL methods runs inside a harness: the code that delays the sampler, assembles each arm's configuration, selects the data for each update and computes each loss. In each of these layers, the quantity that standard logs record can look consistent with a working setup while the quantity that defines the comparison sits one step away. PTH rests on one principle: **measure the quantity that defines the comparison, alongside the quantities that are convenient to log.**

| Layer | Usually logged | Defining quantity | Check | Tool |
|---|---|---|---|---|
| Policy in the PPO ratio | Clip fraction | Ratio to the sampler's recorded probability | 1 | `pth verl`, `ratio_report`, `scan_clip_vs_kl` |
| Configuration of each arm | Run labels | Data seed each arm received | 2 | `pth verl --arm`, `check_arms`, `check_data_orders` |
| Data for each update | Data age | Identity of the training batch | 3 | `pth batches`, `BatchLedger` |
| Implemented loss | A short run trains | Loss equals its written equation | 4 | `check_loss` |

Checks 5 to 7 cover the comparison as a whole (baseline stable range, paired differences per seed, code review against claims). All seven are in [CHECKLIST.md](CHECKLIST.md).

## Install

```bash
pip install git+https://github.com/pth2002/probe-the-harness   # numpy only: checks 1, 2, 3, 6 and the verl log reader

git clone https://github.com/pth2002/probe-the-harness && cd probe-the-harness
pip install ".[all]"     # adds PyTorch for check 4 and PyYAML for YAML configs
```

Python 3.9 or newer.

## Quick start: verl console logs

Point `pth verl` at the console logs of a comparison and assign each log to an arm. Below, the six interval-64 runs from the report: an uncorrected GRPO arm and a truncated importance sampling (TIS) arm, three runs each.

```bash
pth verl --arm "grpo=logs/grpo-*.txt" --arm "tis=logs/tis-*.txt" \
         --vary algorithm.rollout_correction.rollout_is -q
```

```text
logs/grpo-s43.txt  [arm grpo]
  [FAIL] check 1: clip fraction is exactly 0 on all 100 updates while the sampler lags
         Sampler-learner KL reaches 0.313 nat/token (1291x its smallest value) and exceeds 0.05 on 29 updates.
         A clip that never acts under this lag points to a ratio taken against a recomputed policy ...
  [WARN] check 1: the PPO ratio is 1 by construction: pi_old is recomputed and each batch takes one step
         ppo_epochs=1, ppo_mini_batch_size=16, train_batch_size=16, bypass_mode off. Correction for sampler lag: none.

logs/tis-run2.txt  [arm tis]
  passed (2 info hidden)

check 2: configuration of each arm
  [FAIL] check 2: arm 'tis': 3 runs share data.seed
         data.seed=None in tis-run1.txt, tis-run2.txt, tis-run3.txt.
         These runs repeat one data order, so their spread measures sampling noise only.
  [FAIL] check 2: arms use different sets of data.seed
         grpo: ['43', '44', 'None']; tis: ['None']. Results cannot be paired by data.seed across these arms.
```

The other two runs of each arm give the same verdicts. All three GRPO runs fail check 1 and all three TIS runs pass it.

Both findings come from the logs alone. The first is the inactive clip of Section 4.1 of the report and the second is the seed that never reached the TIS arm (Section 4.2). `pth` exits with status 1 when any check fails, so it can gate a results table in CI.

The reader recovers the resolved configuration verl prints at start-up and the `step:N - key:value` lines. It needs `trainer.logger` to include `console` and `actor_rollout_ref.rollout.calculate_log_probs=True`. [docs/verl.md](docs/verl.md) shows how to run check 1 inside the update and how to save configurations and data orders for check 2.

## Python API

```python
import pth

# Check 1: inside the update, with the sampler's recorded log-probabilities.
rep = pth.ratio_report(log_prob, rollout_log_probs, response_mask, logp_old=old_log_prob)
rep.stats["ratio_to_sampler"]["p99"], rep.stats["kl_sampler_to_old"]

# Check 2: resolved configurations of every run, grouped by arm.
rep = pth.check_arms(configs, arm_of=lambda run: run.split("-")[0],
                     vary=["algorithm.rollout_correction.rollout_is"], replicate=["data.seed"])
rep = pth.check_data_orders({run: pth.data_order_fingerprint(first_prompt_ids[run]) for run in runs},
                            seeds=launch_seeds)

# Check 3: which batch each update trains on.
ledger = pth.BatchLedger()
ledger.record(update, pth.batch_fingerprint(batch["input_ids"]), age=age)
print(ledger.report())

# Check 4: the implemented loss against the written equation, value and gradient.
rep = pth.check_loss(my_loss, written_equation, pth.random_batch(), wrt=["logp"])

# Check 6: per-seed paired differences.
rep = pth.paired_differences({"tis": {43: .785, 44: .808}, "grpo": {43: .778, 44: .777}}, "tis", "grpo", scale=100)

rep.raise_on_fail()   # every check returns a Report: findings plus the measured statistics
```

## Examples

```bash
python examples/replay_queue.py       # check 3: a warm-up queue reuses batch 0 for 33 updates at the designed data age
python examples/loss_vs_equation.py   # check 4: a batch token mean against a per-response mean, diagnosed as a normaliser
```

## Reference results on verl

Correctly configured TIS and uncorrected GRPO under sampler lag, from the report. Qwen2.5-Math-1.5B on GSM8K, verl with vLLM 0.11, 16 prompts and 8 responses per update, 512 response tokens, AdamW at 2e-6, KL penalty 1e-3 to the initial model, one optimiser step per batch, 100 updates. The learner's weights reach the sampler on every N-th update only. Final greedy accuracy on the 1319 test problems, by data seed.

| | N = 64, default | 43 | 44 | N = 96, default | 43 | 44 |
|---|---|---|---|---|---|---|
| Uncorrected GRPO | .743 | .778 | .777 | .114 | .280 | .099 |
| TIS | .763 | .785 | .808 | .764 | .781 | .785 |

TIS uses `algorithm.rollout_correction.rollout_is=token` with `rollout_is_threshold=2.0`, which multiplies the loss by the detached weight min(pi_old / q, 2) with q the sampler's recorded probability. TIS stays stable over all 100 updates at both intervals. The uncorrected arm is verl's PPO loss with ppo_epochs=1 and one mini-batch per batch, where the ratio is 1 and the clip cannot act.

## Tests

```bash
pip install ".[all,test]" && pytest
```

## Citation

```bibtex
@misc{pan2026probe,
  title         = {Probe the Harness: Setup Checks for Stale-Data {RL} Comparisons in Language Models},
  author        = {Pan, Taiheng},
  year          = {2026},
  eprint        = {2610.02911},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.02911}
}
```

## License

MIT
