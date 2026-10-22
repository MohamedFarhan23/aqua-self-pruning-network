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

    The per-entry corrections 1 - beta^t are gathered from a lookup table indexed
    by t, not recomputed with an element-wise power every step.
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._num_steps = 0
        self._bc1 = self._bc2 = np.ones(1)
        self._tables: dict[np.dtype, tuple[np.ndarray, np.ndarray]] = {}

    def step(self) -> None:
        self._num_steps += 1  # upper bound on any entry's t
        if len(self._bc1) <= self._num_steps:
            ts = np.arange(2 * self._num_steps + 1)
            # Index 0 (never-active entries, where m = v = 0) maps to 1 to avoid 0/0.
            self._bc1 = np.where(ts == 0, 1.0, 1.0 - self.beta1**ts)
            self._bc2 = np.where(ts == 0, 1.0, 1.0 - self.beta2**ts)
            self._tables.clear()
        super().step()

    def _init_state(self, p, st):
        st["m"] = np.zeros_like(p.data)
        st["v"] = np.zeros_like(p.data)
        st["t"] = np.zeros(p.shape, dtype=np.int64)

    def _update(self, p, st, active):
        g = p.grad
        m, v, t = st["m"], st["v"], st["t"]
        m *= self.beta1
        m += (1 - self.beta1) * g
        v *= self.beta2
        v += (1 - self.beta2) * (g * g)
        if active is None:
            t += 1
        else:
            t += active
            m *= active  # pruned entries: zero state, hence a zero update
            v *= active
        if p.dtype not in self._tables:
            self._tables[p.dtype] = (self._bc1.astype(p.dtype), self._bc2.astype(p.dtype))
        bc1_table, bc2_table = self._tables[p.dtype]
        bc1, bc2 = bc1_table[t], bc2_table[t]
        update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
        if self.weight_decay:
            update += self.weight_decay * p.data
        p.data -= self.lr * update
