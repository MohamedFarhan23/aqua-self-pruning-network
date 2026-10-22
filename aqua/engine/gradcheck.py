"""Finite-difference verification of analytic gradients."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .tensor import Tensor, no_grad


def numerical_grad(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences (O(eps^2) truncation error), perturbing ``x`` in place."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f()
        x[idx] = orig - eps
        minus = f()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: list[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    seed: int = 0,
) -> float:
    """Compare backward() against central differences for every input requiring grad.

    A non-scalar output is contracted with a fixed random tensor R, i.e. we
    check the vector-Jacobian product of L = sum(fn(*inputs) * R). A random R
    (rather than all-ones) catches bugs that cancel under plain summation.
    Returns the largest absolute error; raises AssertionError on mismatch.
    """
    for x in inputs:
        if x.requires_grad and x.dtype != np.float64:
            raise TypeError("gradcheck needs float64 inputs; float32 is too coarse for finite differences")
        x.zero_grad()

    out = fn(*inputs)
    weights = np.random.default_rng(seed).standard_normal(out.shape)
    (out * Tensor(weights)).sum().backward()

    def scalar_loss() -> float:
        with no_grad():
            return float(np.sum(fn(*inputs).data * weights))

    worst = 0.0
    for i, x in enumerate(inputs):
        if not x.requires_grad:
            continue
        analytic = x.grad if x.grad is not None else np.zeros_like(x.data)
        numeric = numerical_grad(scalar_loss, x.data, eps)
        err = np.abs(analytic - numeric)
        if not np.all(err <= atol + rtol * np.abs(numeric)):
            raise AssertionError(
                f"gradient mismatch for input {i} (shape {x.shape}): max abs err {err.max():.3e}\n"
                f"analytic:\n{analytic}\nnumeric:\n{numeric}"
            )
        worst = max(worst, float(err.max()))
    return worst
