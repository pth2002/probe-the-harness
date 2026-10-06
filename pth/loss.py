"""PTH check 4: test every loss against its equation.

A short training run shows that a loss trains. It does not show that the loss
is the one written in the paper. Evaluating the implementation and the written
equation on one fixed input, value and gradient, does. When the gradients agree
up to a per-sequence scale, the two differ in a normaliser.

Requires PyTorch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .report import Report


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("check 4 needs PyTorch: pip install torch") from e
    return torch


def random_batch(
    batch: int = 4,
    length: int = 12,
    seed: int = 0,
    dtype: Optional[Any] = None,
) -> Dict[str, Any]:
    """A fixed, reproducible input for loss tests.

    Returns token log-probabilities under the learner (``logp``), the policy
    used as ``pi_old`` (``logp_old``), the sampler (``logp_sampler``) and a
    reference model (``logp_ref``), one advantage per sequence (``advantages``)
    and a response mask with uneven lengths (``mask``). Values are float64.
    """
    torch = _torch()
    dtype = dtype or torch.float64
    g = torch.Generator().manual_seed(seed)
    logp = -torch.rand(batch, length, generator=g, dtype=dtype) * 4.0 - 0.01
    logp_old = logp - 0.3 * torch.randn(batch, length, generator=g, dtype=dtype)
    logp_sampler = logp - 0.5 * torch.randn(batch, length, generator=g, dtype=dtype)
    logp_ref = logp - 0.4 * torch.randn(batch, length, generator=g, dtype=dtype)
    advantages = torch.randn(batch, generator=g, dtype=dtype)
    lengths = torch.randint(max(1, length // 3), length + 1, (batch,), generator=g)
    mask = (torch.arange(length)[None, :] < lengths[:, None]).to(dtype)
    return {
        "logp": logp.clamp(max=-1e-3),
        "logp_old": logp_old.clamp(max=-1e-3),
        "logp_sampler": logp_sampler.clamp(max=-1e-3),
        "logp_ref": logp_ref.clamp(max=-1e-3),
        "advantages": advantages,
        "mask": mask,
    }


def _scale_diagnosis(torch, g_impl, g_eq, tol: float = 1e-6) -> str:
    a, b = g_impl.reshape(-1), g_eq.reshape(-1)
    na, nb = a.norm(), b.norm()
    if nb == 0 or na == 0:
        return "One of the two gradients is zero."
    cos = float(a @ b / (na * nb))
    if cos > 1 - tol:
        return f"The gradients point the same way and differ by a global factor {float(na / nb):.6g}."
    if g_impl.dim() >= 2:
        rows_a = g_impl.reshape(g_impl.shape[0], -1)
        rows_b = g_eq.reshape(g_eq.shape[0], -1)
        live = rows_b.norm(dim=1) > 0
        if live.any():
            ra, rb = rows_a[live], rows_b[live]
            row_cos = (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-300)
            if bool((row_cos > 1 - tol).all()):
                scale = ra.norm(dim=1) / rb.norm(dim=1)
                return (
                    f"Within every sequence the gradients point the same way, with a per-sequence factor between "
                    f"{float(scale.min()):.4g} and {float(scale.max()):.4g}. The two differ in a normaliser."
                )
    return f"Cosine similarity between the gradients is {cos:.4f}."


def check_loss(
    implemented: Callable[..., Any],
    equation: Callable[..., Any],
    inputs: Mapping[str, Any],
    wrt: Sequence[str] = ("logp",),
    rtol: float = 1e-6,
    atol: float = 1e-9,
    name: str = "loss",
) -> Report:
    """Compare an implemented loss with its written equation, value and gradient.

    Args:
        implemented: the loss as the training code computes it. Called with
            ``**inputs`` and must return a scalar tensor.
        equation: the loss as written in the paper, same signature.
        inputs: a fixed input, for example from :func:`random_batch`.
        wrt: input names to differentiate with respect to.
        rtol, atol: tolerances, applied to the value and to every gradient.
        name: label for the report.
    """
    torch = _torch()
    rep = Report(subject=f"check 4: {name} against its equation")

    def run(fn):
        args = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.detach().clone()
                if k in wrt:
                    v.requires_grad_(True)
            args[k] = v
        out = fn(**args)
        if not torch.is_tensor(out) or out.numel() != 1:
            raise ValueError(f"{getattr(fn, '__name__', fn)} must return a scalar tensor")
        grads = torch.autograd.grad(out, [args[k] for k in wrt], allow_unused=True)
        return out.detach(), {k: (g if g is not None else torch.zeros_like(args[k])) for k, g in zip(wrt, grads)}

    v_impl, g_impl = run(implemented)
    v_eq, g_eq = run(equation)
    rep.stats["value_implemented"] = float(v_impl)
    rep.stats["value_equation"] = float(v_eq)

    value_ok = bool(torch.allclose(v_impl, v_eq.to(v_impl.dtype), rtol=rtol, atol=atol))
    if not value_ok:
        rep.add(4, "fail", f"{name}: value differs from the equation",
                f"implemented {float(v_impl):.8g}, equation {float(v_eq):.8g}")

    all_ok = value_ok
    for k in wrt:
        a, b = g_impl[k], g_eq[k].to(g_impl[k].dtype)
        diff = float((a - b).abs().max())
        rep.stats[f"max_grad_diff[{k}]"] = diff
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            all_ok = False
            rep.add(4, "fail", f"{name}: gradient with respect to {k} differs from the equation",
                    f"max abs difference {diff:.3g}. " + _scale_diagnosis(torch, a, b))
    if all_ok:
        rep.add(4, "info", f"{name}: value and gradient match the equation",
                f"value {float(v_impl):.8g}, gradients with respect to {', '.join(wrt)}")
    return rep
