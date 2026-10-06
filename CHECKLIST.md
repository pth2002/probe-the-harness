# The PTH checklist

**Principle.** For every layer of the harness, measure the quantity that defines the comparison, alongside the quantities that are convenient to log.

Checks 1 to 4 apply the principle to the four layers of a stale-data harness. Checks 5 to 7 apply it to the comparison as a whole.

| # | Check | Signal | Tool |
|---|---|---|---|
| 1 | **Log the ratio against the policy that generated the data.** For every arm, record the clip fraction and the distribution of pi_theta / q, with q the sampler's recorded probability. | Under a lagged sampler, a clip fraction of exactly zero next to a sampler-learner KL that grows by orders of magnitude. The source of pi_old in the code settles it. | `pth verl`, `ratio_report`, `scan_clip_vs_kl` |
| 2 | **Check the resolved configuration per arm.** Print the full configuration each run used. Confirm per arm that the fields the comparison varies differ and the fields it holds fixed match, data seed included. | Runs labelled with different seeds that share one data order. Arms that ran different sets of seeds. | `pth verl --arm`, `check_arms`, `check_data_orders` |
| 3 | **Log the identity of the training batch alongside its age.** In a replay simulator, count the distinct batches used and confirm that the count matches the design. | Fewer distinct batches than updates when the design calls for a new batch at every update, even with a data age that matches the design. | `pth batches`, `BatchLedger` |
| 4 | **Test every loss against its equation.** For each variant, evaluate the implemented loss and its gradient on a fixed input and compare them with the written formula. | Values or gradients that differ. Gradients that agree up to a per-sequence factor point to a different normaliser. | `check_loss` |
| 5 | **Map each baseline's stable range.** Before comparing, find where the baseline degrades under its intended configuration. | A baseline that degrades in a range where it is expected to hold. Confirm checks 1 to 4 for that arm before attributing the drop to the method. | procedure |
| 6 | **Report paired differences per seed.** Pooled rankings mix seed effects with method effects. Per-seed pairs separate them. | Seeds that ran in one arm only. Paired differences that change sign. | `paired_differences`, `pth paired` |
| 7 | **Review code and logs against each claim.** Read launch scripts and logs line by line against the sentences of the draft. | A sentence in the draft that no line of code or log supports. | procedure |

## How each check caught a detail in the report

1. **The Idle Clip.** verl recomputes pi_old with the learner unless bypass mode is on. With one epoch and one mini-batch per batch, the ratio is 1 on every token. Across all 21 lagged baseline runs, the clip fraction was zero on every update while the sampler-learner KL reached 7 to 9 nats per token at refresh interval 96.
2. **The Lost Seed.** The launch script appended the data seed to a variable and then, when importance sampling was enabled, assigned a new string to the same variable. The three TIS runs repeated one data order.
3. **The Stuck Batch.** The replay queue trained on the oldest batch available until it held k + 1 batches. At replay age 32, the first 33 updates used batch 0, and 100 updates used 68 distinct batches, with the data age on schedule throughout.
4. **The Rogue Normaliser.** Two loss variants divided by a different normaliser than the one written for them. Both trained normally.

## A review template for check 7

| Claim in the draft | Code that produces it | Log or file that shows it | Checked by |
|---|---|---|---|
| | | | |
