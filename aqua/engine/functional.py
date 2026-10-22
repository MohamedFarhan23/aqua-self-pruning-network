"""Fused, numerically stable ops whose naive composition would overflow."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    # log softmax(x) = (x - c) - log sum exp(x - c) for any c; c = max(x) keeps
    # every exponent <= 0, so exp() cannot overflow and the sum is >= 1.
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    out = _log_softmax(x.data, axis)
    softmax = np.exp(out)

    def backward(g):
        # d/dx_j of (x_i - lse(x)) = delta_ij - softmax_j
        return (g - softmax * g.sum(axis=axis, keepdims=True),)

    return Tensor._make(out, (x,), backward, "log_softmax")


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis).exp()


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy of ``logits`` (N, C) against integer labels (N,).

    Fused so the backward pass is the closed form (softmax - onehot) / N, which
    never forms a log of a probability that underflowed to zero.
    """
    targets = np.asarray(targets)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError(f"expected logits (N, C) and targets (N,), got {logits.shape} and {targets.shape}")
    n = logits.shape[0]
    rows = np.arange(n)
    logp = _log_softmax(logits.data, axis=1)
    loss = -logp[rows, targets].mean()

    def backward(g):
        grad = np.exp(logp)
        grad[rows, targets] -= 1.0
        return (grad * (g / n),)

    return Tensor._make(np.asarray(loss, dtype=logits.dtype), (logits,), backward, "cross_entropy")
