from __future__ import annotations

from typing import Iterable

import numpy as np

from ..engine.parameter import Parameter
from .optimizer import Optimizer


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with decoupled weight decay (AdamW) and per-entry step counts.

    Bias correction divides m by (1 - beta1^t) because m is a zero-initialised
    EMA of length t. A revived entry's EMA restarts at zero, so its t must
    restart too. With a shared global t = T, the first revived step would be
    lr * (1 - beta1) / (1 - beta1^T) * sqrt((1 - beta2^T) / (1 - beta2)) * sign(g),
    i.e. 2.5x lr at T = 1000 and 3.16x lr as T grows (default betas), instead of
    lr * sign(g). That is why ``t`` is an array here.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def _init_state(self, p, st):
        st["m"] = np.zeros_like(p.data)
        st["v"] = np.zeros_like(p.data)
        st["t"] = np.zeros(p.shape, dtype=np.int64)

    def _update(self, p, st, active):
        g = p.grad
        m, v, t = st["m"], st["v"], st["t"]
        t += active
        m[...] = np.where(active, self.beta1 * m + (1 - self.beta1) * g, 0.0)
        v[...] = np.where(active, self.beta2 * v + (1 - self.beta2) * g * g, 0.0)
        steps = np.maximum(t, 1)  # pruned entries have t = 0 and m = 0, so they get a zero update
        m_hat = m / (1 - self.beta1**steps)
        v_hat = v / (1 - self.beta2**steps)
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        if self.weight_decay:
            update = update + self.weight_decay * p.data
        p.data -= self.lr * update
