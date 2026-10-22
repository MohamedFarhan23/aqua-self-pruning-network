"""Optimizer base class with mask-aware per-entry state.

Why a masked weight needs optimizer support, not just a zero gradient:
the engine already gives pruned entries grad == 0, but momentum-style
optimizers do not stop on a zero gradient. Adam with g = 0 still applies
lr * m_hat / (sqrt(v_hat) + eps) from the decaying history, so a pruned weight
keeps drifting. If it is later revived, it returns with a value it never
earned and stale moments from before it was pruned.

The rule enforced here: on every step, a pruned entry has zero state and zero
value, and an entry that was pruned at any point since the previous step
restarts from fresh state. All state is per-entry, including Adam's step count,
so bias correction for a revived entry starts again at t = 1.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..engine.parameter import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        if any(not isinstance(p, Parameter) for p in self.params):
            raise TypeError("optimizers only accept Parameter instances")
        self.lr = lr
        self.state: list[dict[str, np.ndarray]] = [{} for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def state_for(self, p: Parameter) -> dict[str, np.ndarray]:
        for q, st in zip(self.params, self.state):
            if q is p:
                return st
        raise KeyError("parameter not managed by this optimizer")

    def step(self) -> None:
        for p, st in zip(self.params, self.state):
            dropped = p.consume_dropped()
            if p.grad is None:
                if dropped is not None:  # keep the reset pending until we actually step
                    p._dropped = dropped
                continue
            if not st:
                self._init_state(p, st)
            active = p.mask if p.mask is not None else np.ones(p.shape, dtype=bool)
            reset = ~active if dropped is None else ~active | dropped
            for buf in st.values():
                buf[reset] = 0
            self._update(p, st, active)
            # Invariant 1. The update is already zero on pruned entries; this
            # line makes it a guarantee rather than a floating-point accident.
            p.data[~active] = 0.0

    def _init_state(self, p: Parameter, st: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def _update(self, p: Parameter, st: dict[str, np.ndarray], active: np.ndarray) -> None:
        raise NotImplementedError
