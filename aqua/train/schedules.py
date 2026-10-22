"""Learning-rate schedules: functions from optimizer step (1-based) to learning rate."""

from __future__ import annotations

import math
from typing import Callable

Schedule = Callable[[int], float]


def constant(lr: float) -> Schedule:
    return lambda step: lr


def cosine(lr: float, total_steps: int, final_fraction: float = 0.0) -> Schedule:
    """Cosine decay from ``lr`` to ``final_fraction * lr`` over ``total_steps``.

    Adam's step size is roughly lr per coordinate regardless of gradient scale,
    so at a constant lr the iterate keeps bouncing around a minimum with
    amplitude ~lr. Decaying lr to ~0 lets it settle, and the late low-lr phase
    is also where a pruned network recovers from its last pruning step.
    """
    def schedule(step: int) -> float:
        progress = min(step / total_steps, 1.0)
        return lr * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))
    return schedule
