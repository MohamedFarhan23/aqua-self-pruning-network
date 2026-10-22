"""Gradual sparsity schedule (Zhu & Gupta, 2017).

    s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t1 - t0))^3   for t0 <= t <= t1

ds/dt is largest at t0 and falls to 0 at t1. Early on the dense network is
highly redundant and recovers quickly, so it can lose many connections per
event. Near the target every remaining connection matters more, and each
pruning event is a larger perturbation relative to what is left, so the
schedule slows down and leaves time to recover. Small steps also keep each
event inside the regime where the diagonal Taylor estimate is valid
(see ``criteria.py``).

One-shot pruning at the end (prune once, then fine-tune) is the alternative.
It is cheaper, but the scores are computed once on the dense network and
never revisited, and the network has to absorb the whole perturbation at
once. The gradual schedule re-scores after every slice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubicSchedule:
    final_sparsity: float
    start_step: int
    end_step: int
    interval: int = 100
    initial_sparsity: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.initial_sparsity <= self.final_sparsity < 1.0:
            raise ValueError("need 0 <= initial_sparsity <= final_sparsity < 1")
        if not 0 < self.start_step <= self.end_step:
            raise ValueError("need 0 < start_step <= end_step")
        if self.interval <= 0:
            raise ValueError("interval must be positive")

    def sparsity(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return self.final_sparsity
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3

    def progress(self, step: int) -> float:
        return min(max((step - self.start_step) / max(self.end_step - self.start_step, 1), 0.0), 1.0)

    def is_event(self, step: int) -> bool:
        """Pruning happens every ``interval`` steps from start to end, and always at end."""
        if step == self.end_step:
            return True
        return self.start_step <= step < self.end_step and (step - self.start_step) % self.interval == 0
