"""PTH check 4: an implemented loss against the equation written for it.

The written loss averages token terms within each response and then across
responses. The implementation below averages over all tokens in the batch.
Both train, and their gradients point the same way within every response, so
a short run shows nothing unusual. The check reports the per-response factor
that separates them.

    python examples/loss_vs_equation.py
"""
import torch

from pth import check_loss, random_batch


def tis_written(logp, logp_old, logp_sampler, advantages, mask, **_):
    """L = -1/B sum_i 1/|y_i| sum_t min(pi_old/q, 2) * A_i * log pi_theta(y_it)."""
    w = torch.clamp(torch.exp(logp_old - logp_sampler), max=2.0).detach()
    per_token = -w * advantages[:, None] * logp * mask
    return (per_token.sum(1) / mask.sum(1)).mean()


def tis_implemented(logp, logp_old, logp_sampler, advantages, mask, **_):
    w = torch.clamp(torch.exp(logp_old - logp_sampler), max=2.0).detach()
    per_token = -w * advantages[:, None] * logp * mask
    return per_token.sum() / mask.sum()               # token mean over the whole batch


def tis_fixed(logp, logp_old, logp_sampler, advantages, mask, **_):
    w = torch.clamp(torch.exp(logp_old - logp_sampler), max=2.0).detach()
    per_token = -w * advantages[:, None] * logp * mask
    return (per_token.sum(1) / mask.sum(1)).mean()   # mean within each response, then across


if __name__ == "__main__":
    batch = random_batch(batch=4, length=12, seed=0)
    print(check_loss(tis_implemented, tis_written, batch, name="TIS, batch token mean"))
    print()
    print(check_loss(tis_fixed, tis_written, batch, name="TIS, response mean"))
