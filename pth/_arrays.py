"""Small array helpers so the checks accept numpy arrays, lists or torch tensors."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("expected an array, got None")
    if hasattr(x, "detach"):  # torch.Tensor without importing torch
        x = x.detach()
        if hasattr(x, "float"):
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def masked(x: Any, mask: Optional[Any]) -> np.ndarray:
    """Return the entries of ``x`` where ``mask`` is nonzero, flattened."""
    arr = as_numpy(x)
    if mask is None:
        return arr.reshape(-1)
    m = as_numpy(mask)
    if m.shape != arr.shape:
        raise ValueError(f"mask shape {m.shape} does not match values shape {arr.shape}")
    return arr[m > 0]
