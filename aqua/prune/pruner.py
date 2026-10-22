"""The self-pruning controller: called after every optimizer step by the training loop.

At each event of the schedule, the pruner ranks every connection of every layer
together (global, not per layer) and keeps exactly the top-k, where
k = round((1 - s(t)) * total connections). The budget is hard: after the last
event exactly k connections are active, and the rest are exact zeros removed
from the forward pass by their mask.

A global ranking is meaningful because the scores are estimates of loss change
in nats, comparable across layers. Magnitudes are not: in LeNet-300-100, fc1
weights start ~4x smaller than fc3's (std sqrt(1/784) vs sqrt(2/100)), for
reasons unrelated to importance. The magnitude baseline is still ranked globally here,
so both criteria get the same allocation freedom and differ only in scoring.

Regrowth (optional)
-------------------
Active connection i has value "removal cost" = score_i, and pruned connection j
has value "regrowth gain" = g_j^2 / 2F_j (criteria.py). Both are in nats, so one
top-k over the union decides which to keep, which to drop and which to revive.
Swaps happen only when a pruned connection is predicted to be worth more than an
active one. Stability measures:

* Revived weights start at exactly 0, so the network computes the same function
  immediately after the event (a pure mask change cannot spike the loss). Their
  optimizer state starts fresh (Part 1 guarantees), so the first update is the
  normal ~lr Adam step, not a stale-momentum jump.
* A weight revived at 0 would have removal score 0 and be re-pruned at the next
  event before it could grow. Revived connections are protected for ``grace``
  steps.
* At most ``regrow_fraction * k`` connections can be revived per event, and this
  cap decays with a cosine over the ramp and reaches 0 at its end. The mask
  therefore freezes for the final fine-tuning phase, and the network converges
  on a fixed architecture.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from ..engine import Parameter, Tensor, cross_entropy
from ..nn import MLP, Linear
from .criteria import CRITERIA, NEEDS_STATS, Stats, per_example_fisher, regrowth_gain


def criterion_needs_factors(name: str) -> bool:
    return name == "obs"
from .schedule import CubicSchedule


def batch_stats(model: MLP, x: np.ndarray, y: np.ndarray, chunk: int = 1024) -> list[Stats]:
    """Gradient and per-example Fisher of every weight matrix at the current weights, over (x, y).

    Each chunk's per-example Fisher is the mean of squared per-example
    gradients within the chunk, so a size-weighted average over chunks is the
    mean over all of (x, y). Leaves every .grad cleared.
    """
    layers = model.linears
    previous = [layer.capture for layer in layers]
    grads = [np.zeros(layer.weight.shape) for layer in layers]
    fishers = [np.zeros(layer.weight.shape) for layer in layers]
    a_factors = [np.zeros((layer.weight.shape[0],) * 2) for layer in layers]
    g_factors = [np.zeros((layer.weight.shape[1],) * 2) for layer in layers]
    for layer in layers:
        layer.capture = True
    try:
        for start in range(0, len(y), chunk):
            xb, yb = x[start:start + chunk], y[start:start + chunk]
            for p in model.parameters():
                p.zero_grad()
            cross_entropy(model(Tensor(xb)), yb).backward()
            frac = len(yb) / len(y)
            for i, layer in enumerate(layers):
                grads[i] += layer.weight.dense_grad * frac
                fishers[i] += per_example_fisher(layer.input, layer.output_grad) * frac
                x64 = layer.input.astype(np.float64)
                u = layer.output_grad.astype(np.float64) * len(yb)  # per-example dl_n/dy_n
                a_factors[i] += x64.T @ x64 / len(y)
                g_factors[i] += u.T @ u / len(y)
    finally:
        for layer, prev in zip(layers, previous):
            layer.capture = prev
            layer.input = layer.output_grad = None
        for p in model.parameters():
            p.zero_grad()
    return [Stats(g, f, a, gf) for g, f, a, gf in zip(grads, fishers, a_factors, g_factors)]


@dataclass
class PruneEvent:
    step: int
    target_sparsity: float
    sparsity: float
    pruned: int  # active -> pruned at this event
    revived: int  # pruned -> active at this event
    layer_density: dict[str, float] = field(default_factory=dict)


class Pruner:
    def __init__(
        self,
        model: MLP,
        schedule: CubicSchedule,
        criterion: str = "obs",
        criterion_kwargs: dict | None = None,
        ema: float = 0.99,
        regrow: bool = False,
        regrow_fraction: float = 0.05,
        grace: int | None = None,
        calibration: tuple[np.ndarray, np.ndarray] | None = None,
        calibration_size: int = 4096,
        seed: int = 0,
    ):
        if criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {sorted(CRITERIA)}")
        self.model = model
        self.layers: list[Linear] = model.linears
        self.params: list[Parameter] = [layer.weight for layer in self.layers]
        self.schedule = schedule
        self.criterion = criterion
        self.score_fn = functools.partial(CRITERIA[criterion], **(criterion_kwargs or {}))
        self.ema = ema
        self.regrow = regrow
        self.regrow_fraction = regrow_fraction
        self.grace = 2 * schedule.interval if grace is None else grace
        self.total = sum(p.size for p in self.params)
        self.events: list[PruneEvent] = []

        self.needs_stats = criterion in NEEDS_STATS or regrow
        # "batch": g and F recomputed at the current weights on a fresh calibration
        # batch at every event. "ema": running averages over training steps (stale;
        # kept as an ablation, see DESIGN.md).
        self.stats_source = "batch" if calibration is not None else "ema"
        self.calibration = calibration
        self.calibration_size = calibration_size
        self._rng = np.random.default_rng(seed)
        self.stats: list[Stats | None] = [None] * len(self.params)
        self.revived_at = [np.full(p.shape, -(10**12), dtype=np.int64) for p in self.params]
        for p in self.params:
            if p.mask is None:
                p.set_mask(np.ones(p.shape, dtype=bool))  # so dense_grad is tracked
        self._set_capture(self.needs_stats and self.stats_source == "ema")

    # ------------------------------------------------------------- reporting
    @property
    def num_active(self) -> int:
        return sum(p.num_active for p in self.params)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.num_active / self.total

    def layer_density(self) -> dict[str, float]:
        return {p.name: 1.0 - p.sparsity for p in self.params}

    # ---------------------------------------------------------------- hooks
    def on_step(self, step: int) -> None:
        """Training-loop hook, run after the optimizer step while grads are still populated."""
        if self.needs_stats and self.stats_source == "ema" and step <= self.schedule.end_step:
            self._accumulate()
        if self.schedule.is_event(step):
            self._prune(step)
        if step >= self.schedule.end_step:
            self._set_capture(False)  # mask is final; stop paying for statistics

    def _set_capture(self, on: bool) -> None:
        for layer in self.layers:
            layer.capture = on
            if not on:
                layer.input = layer.output_grad = None

    def _accumulate(self) -> None:
        for i, (layer, p) in enumerate(zip(self.layers, self.params)):
            if p.dense_grad is None or layer.output_grad is None:
                continue
            grad = p.dense_grad.astype(np.float64)
            fisher = per_example_fisher(layer.input, layer.output_grad)
            st = self.stats[i]
            if st is None:
                self.stats[i] = Stats(grad, fisher)
            else:
                st.grad *= self.ema
                st.grad += (1 - self.ema) * grad
                st.fisher *= self.ema
                st.fisher += (1 - self.ema) * fisher

    # ---------------------------------------------------------------- prune
    def _prune(self, step: int) -> None:
        target = self.schedule.sparsity(step)
        k = int(round((1.0 - target) * self.total))
        if self.needs_stats and self.stats_source == "batch":
            x, y = self.calibration
            idx = self._rng.choice(len(y), size=min(self.calibration_size, len(y)), replace=False)
            self.stats = batch_stats(self.model, x[idx], y[idx])
        if criterion_needs_factors(self.criterion) and self.stats_source != "batch":
            raise ValueError(f"{self.criterion} needs calibration-batch statistics (pass calibration=...)")
        if self.needs_stats and any(st is None for st in self.stats):
            raise RuntimeError("pruning event before any gradient statistics were collected")

        values = []
        for i, p in enumerate(self.params):
            active = p.mask
            v = np.full(p.shape, -np.inf)
            v[active] = self.score_fn(p.data, self.stats[i])[active]
            protected = active & (step - self.revived_at[i] < self.grace)
            v[protected] = np.inf
            values.append(v)

        if self.regrow:
            self._add_regrowth_candidates(values, k, step)

        flat = np.concatenate([v.ravel() for v in values])
        if k > np.count_nonzero(flat > -np.inf):
            raise RuntimeError("sparsity target decreased below the current mask without regrowth")
        keep = np.zeros(flat.size, dtype=bool)
        if k > 0:
            # Stable sort with index tie-break: the same scores always give the same mask.
            keep[np.argsort(-flat, kind="stable")[:k]] = True

        pruned = revived = offset = 0
        for i, p in enumerate(self.params):
            new = keep[offset:offset + p.size].reshape(p.shape)
            offset += p.size
            old = p.mask
            newly_revived = new & ~old
            pruned += int(np.count_nonzero(old & ~new))
            revived += int(np.count_nonzero(newly_revived))
            self.revived_at[i][newly_revived] = step
            p.set_mask(new)
            assert np.all(p.data[~new] == 0.0)

        self.events.append(PruneEvent(step, target, self.sparsity, pruned, revived, self.layer_density()))

    def _add_regrowth_candidates(self, values: list[np.ndarray], k: int, step: int) -> None:
        """Give the most promising pruned connections a value (their predicted regrowth gain)."""
        progress = self.schedule.progress(step)
        cap = int(self.regrow_fraction * k * 0.5 * (1 + math.cos(math.pi * progress)))
        if cap == 0:
            return
        gains = [np.where(p.mask, -np.inf, regrowth_gain(self.stats[i])) for i, p in enumerate(self.params)]
        flat = np.concatenate([g.ravel() for g in gains])
        # Exactly the top `cap` by gain, and only those with a strictly positive gain:
        # a zero gain (e.g. a connection into a dead ReLU unit, whose gradient is
        # identically 0) would be revived at 0 and could never move.
        top = np.argsort(-flat, kind="stable")[:cap]
        top = top[flat[top] > 0]
        chosen = np.zeros(flat.size, dtype=bool)
        chosen[top] = True
        offset = 0
        for v, g in zip(values, gains):
            c = chosen[offset:offset + g.size].reshape(g.shape)
            offset += g.size
            v[c] = g[c]
