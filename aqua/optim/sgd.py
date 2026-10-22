from __future__ import annotations

from typing import Iterable

import numpy as np

from ..engine.parameter import Parameter
from .optimizer import Optimizer


class SGD(Optimizer):
    """SGD with (heavy-ball) momentum and L2 weight decay, restricted to active entries."""

    def __init__(self, params: Iterable[Parameter], lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def _init_state(self, p, st):
        st["velocity"] = np.zeros_like(p.data)

    def _update(self, p, st, active):
        g = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
        v = st["velocity"]
        v[...] = np.where(active, self.momentum * v + g, 0.0)
        p.data -= self.lr * v
