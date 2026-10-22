"""Importance criteria: estimates of how much the loss rises if a connection is removed.

Setting up the estimate
-----------------------
Let L(w) be the mean training loss. Removing connection i sets w_i -> 0, a
perturbation dw = -w_i e_i. Taylor-expanding to second order:

    dL_i = L(w - w_i e_i) - L(w) ~ -g_i w_i + 1/2 H_ii w_i^2          (*)

with g = dL/dw and H the Hessian. The criteria below approximate (*) in
different ways:

* ``magnitude``: |w_i|. Implicitly assumes g_i = 0 and the same H_ii for every
  connection. That is false across layers (fan-in and activation scale differ)
  and within a layer.
* ``taylor1``: |g_i w_i|, the first-order term only (Molchanov et al., 2017).
* ``obd``: 1/2 F_ii w_i^2, the second-order term only (Optimal Brain Damage,
  LeCun et al., 1990, with the Fisher in place of the Hessian).
* ``taylor2``: -g_i w_i + 1/2 F_ii w_i^2, all of (*). With exact statistics it is
  by far the best predictor of the loss change from removing ONE weight
  (results/part3/criterion_check.json). For SETS it fails badly: summed over a
  layer, -sum_i g_i w_i is the derivative of the loss w.r.t. scaling the whole
  layer, a large coherent signal. The diagonal model then predicts that removing
  most of a layer lowers the loss, and the pruner collapses whole layers
  (tests/test_pruning.py::test_taylor2_collapses_layers_when_pruning_sets).
  It is kept as a documented negative result.
* ``obs`` (ours, the default): the Optimal Brain Surgeon cost with a
  Kronecker-factored Fisher. See below.

Why OBS is the right target here
--------------------------------
(*) is the cost of removing w_i while nothing else changes. But this pruner
retrains between events, so the network gets to compensate. The cost that
survives retraining is, to second order, the OBS cost (Hassibi & Stork, 1993):

    min over dw with dw_q = -w_q of  1/2 dw^T H dw   =   w_q^2 / (2 [H^-1]_qq)

This is evaluated at a stationary point (g ~ 0), which retraining keeps us near.
That is also why the first-order term is dropped: a first-order gain is
something the optimizer collects anyway. The two differ most where inputs are
correlated. 68 of MNIST's 784 pixels are constant over the training split,
hence perfectly redundant with one another (and with the bias): OBD and magnitude must pay for each such weight separately, while
OBS knows that any one of them can be removed and absorbed by its twins
(tests/test_pruning.py::test_obs_compensation_lowers_cost_for_correlated_inputs).

H^-1 for the whole network is intractable. For a linear layer, the Fisher of
W is E[(x x^T) (x) (u u^T)] with u = dl_n/dy. K-FAC (Martens & Grosse, 2015)
approximates it as A (x) G with A = E[x x^T] and G = E[u u^T], whose inverse is
A^-1 (x) G^-1. So [F^-1]_(ij),(ij) = [A^-1]_ii [G^-1]_jj, and a 784x784 and a
300x300 inverse per event give every score. The approximations: the Fisher is
block-diagonal per layer (no compensation across layers), it is Kronecker
factored, and compensation may use weights that are already pruned (the
pruned set is not removed from A). All three are measured rather than assumed
in Part 4: OBS has to beat magnitude in final accuracy to justify itself.

Where F comes from
------------------
H_ii for a deep net is not available in closed form. For a per-example loss
l_n = -log p(y_n | x_n), the Gauss-Newton matrix approximates the Hessian, and
for softmax cross-entropy it equals the Fisher information. We estimate its
diagonal with the per-example empirical Fisher, F_ii = (1/N) sum_n (dl_n/dw_i)^2.

This needs *per-example* squared gradients: the square of the mini-batch mean
gradient is not the same thing (it is smaller by a factor of ~N where gradients
are noisy). For a linear layer y = x W, with delta_n = dL/dy_n being the
gradient of the batch-mean loss L = (1/N) sum_n l_n:

    dl_n/dW_ij = N x_ni delta_nj
    F_ij = (1/N) sum_n N^2 x_ni^2 delta_nj^2 = N * [(x^2)^T (delta^2)]_ij

That is one extra matmul per layer (``per_example_fisher``).

Where g and F are measured: the first version kept exponential moving averages
over training steps. They are stale: they mix gradients from many past weight
positions while Adam moves the weights, and with them taylor2's prediction
quality fell from Spearman 0.95 to 0.06 (criterion_check.json). The pruner now
recomputes all statistics at the current weights, on a fresh 4,096-example
calibration batch at each event (``Pruner(calibration=...)``). EMA statistics
remain available as an ablation.

Removing many weights at once still breaks any per-weight score, OBS included
(OBS compensates for one removal at a time). That is the reason for pruning
gradually (``schedule.py``): each event removes a small slice, and training in
between re-equilibrates before the next one.

Regrowth
--------
For a pruned connection (w_i = 0), the same quadratic model gives the value of
bringing it back: L(w_i) - L(0) ~ g_i w_i + 1/2 F_ii w_i^2 is minimised at
w_i* = -g_i / F_ii, with predicted loss decrease g_i^2 / (2 F_ii). That is in
the same units (nats of loss) as the removal cost of an active weight, so
growing and pruning can be decided by one global ranking.

All scores are computed in float64: F w^2 is small enough to lose precision in
float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Stats:
    """Running gradient statistics for one weight matrix (all arrays shaped like it)."""

    grad: np.ndarray  # dL/d(w*m): mean gradient, defined for pruned entries too
    fisher: np.ndarray  # per-example empirical Fisher diagonal
    # Kronecker factors of the layer's Fisher, F ~ A (x) G (calibration batches only):
    a: np.ndarray | None = None  # E[x x^T] over inputs, (in, in)
    g: np.ndarray | None = None  # E[u u^T] over per-example output gradients u = dl_n/dy, (out, out)


Criterion = Callable[[np.ndarray, Stats], np.ndarray]


def magnitude(w: np.ndarray, st: Stats | None) -> np.ndarray:
    return np.abs(w).astype(np.float64)


def taylor1(w: np.ndarray, st: Stats) -> np.ndarray:
    return np.abs(st.grad * w)


def obd(w: np.ndarray, st: Stats) -> np.ndarray:
    return 0.5 * st.fisher * w.astype(np.float64) ** 2


def taylor2(w: np.ndarray, st: Stats) -> np.ndarray:
    w = w.astype(np.float64)
    return 0.5 * st.fisher * w**2 - st.grad * w


OBS_DAMPING = 0.1


def obs(w: np.ndarray, st: Stats, damping: float | None = None) -> np.ndarray:
    """Optimal Brain Surgeon cost with a Kronecker-factored Fisher (K-FAC).

    OBS (Hassibi & Stork, 1993): removing w_q, with every other weight moved
    optimally to compensate, costs w_q^2 / (2 [H^-1]_qq) to second order. For a
    linear layer the Fisher factorises approximately as F = A (x) G, so
    [F^-1]_(ij),(ij) = [A^-1]_ii [G^-1]_jj, and two small inverses give every score.

    Damping: both factors get Tikhonov damping proportional to their mean
    eigenvalue, lambda = damping * tr(.)/dim. MNIST has constant border pixels,
    so A is singular without it. The default comes from a validation-set check,
    not the test set.
    """
    if st.a is None or st.g is None:
        raise ValueError("obs needs Kronecker factors: use calibration-batch statistics")
    lam = OBS_DAMPING if damping is None else damping

    def inv_diag(m: np.ndarray) -> np.ndarray:
        d = m.shape[0]
        return np.diag(np.linalg.inv(m + lam * np.trace(m) / d * np.eye(d)))

    w = w.astype(np.float64)
    return w**2 / (2.0 * np.outer(inv_diag(st.a), inv_diag(st.g)))


CRITERIA: dict[str, Criterion] = {
    "magnitude": magnitude,
    "taylor1": taylor1,
    "obd": obd,
    "taylor2": taylor2,
    "obs": obs,
}
NEEDS_STATS = {"taylor1", "obd", "taylor2", "obs"}


def regrowth_gain(st: Stats, damping: float = 1e-12) -> np.ndarray:
    """Predicted loss decrease from reviving a pruned weight and optimising it: g^2 / 2F."""
    return st.grad**2 / (2.0 * (st.fisher + damping))


def per_example_fisher(x: np.ndarray, output_grad: np.ndarray) -> np.ndarray:
    """Diagonal empirical Fisher of W for y = x W, from the layer input and dL/dy of the batch mean."""
    n = x.shape[0]
    x64, d64 = x.astype(np.float64), output_grad.astype(np.float64)
    return n * ((x64 * x64).T @ (d64 * d64))
