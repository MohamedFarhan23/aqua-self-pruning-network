"""Mini-batch training and evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ..data import Dataset, Split
from ..engine import Tensor, cross_entropy, no_grad
from ..nn import Module
from ..optim import Optimizer
from .schedules import Schedule


@dataclass
class EpochStats:
    epoch: int
    step: int
    train_loss: float  # running mean over the epoch's mini-batches
    train_acc: float
    val_loss: float
    val_acc: float
    max_grad_norm: float  # largest global gradient L2 norm seen this epoch
    seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(model: Module, split: Split, batch_size: int = 4096) -> tuple[float, float]:
    """Mean cross-entropy and accuracy over a split, without building a graph."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(split), batch_size):
            x, y = split.x[start:start + batch_size], split.y[start:start + batch_size]
            logits = model(Tensor(x))
            total_loss += cross_entropy(logits, y).item() * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
    return total_loss / len(split), correct / len(split)


def global_grad_norm(model: Module) -> float:
    return float(np.sqrt(sum(float(np.sum(p.grad.astype(np.float64) ** 2))
                             for p in model.parameters() if p.grad is not None)))


def train(
    model: Module,
    optimizer: Optimizer,
    data: Dataset,
    epochs: int,
    batch_size: int = 128,
    seed: int = 0,
    lr_schedule: Schedule | None = None,
    on_step: Callable[[int], None] | None = None,
    on_epoch: Callable[[EpochStats], None] | None = None,
) -> list[EpochStats]:
    """Shuffled mini-batch training.

    ``lr_schedule(step)`` sets the optimizer's learning rate before each step.
    ``on_step(step)`` runs after every optimizer step (the hook the pruner
    uses). Training stops with FloatingPointError on the first non-finite loss
    or gradient, so a reported run is guaranteed NaN-free.
    """
    rng = np.random.default_rng(seed)
    n = len(data.train)
    history: list[EpochStats] = []
    step = 0
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        perm = rng.permutation(n)
        loss_sum, correct, max_norm = 0.0, 0, 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x, y = data.train.x[idx], data.train.y[idx]

            logits = model(Tensor(x))
            loss = cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()

            norm = global_grad_norm(model)
            if not (np.isfinite(loss.item()) and np.isfinite(norm)):
                raise FloatingPointError(f"non-finite loss/gradient at epoch {epoch}, step {step}")
            if lr_schedule is not None:
                optimizer.lr = lr_schedule(step + 1)
            optimizer.step()
            step += 1
            if on_step is not None:
                on_step(step)

            loss_sum += loss.item() * len(idx)
            correct += int((logits.data.argmax(axis=1) == y).sum())
            max_norm = max(max_norm, norm)

        val_loss, val_acc = evaluate(model, data.val)
        stats = EpochStats(epoch, step, loss_sum / n, correct / n, val_loss, val_acc, max_norm,
                           time.perf_counter() - start_time)
        history.append(stats)
        if on_epoch is not None:
            on_epoch(stats)
    return history
