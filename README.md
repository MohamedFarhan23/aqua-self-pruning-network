# AQUA — The Self-Pruning Network

A neural-network stack written from scratch in Python + NumPy: a reverse-mode
autodiff engine, mask-aware optimizers, and (in later parts) a classifier that
prunes its own connections during training.

No PyTorch / TensorFlow / JAX / autodiff libraries are used. Only NumPy is a
runtime dependency.

## Status

| Part | Scope | State |
| --- | --- | --- |
| 1 | Autodiff engine, gradient checks, masked-weight correctness | done |
| 2 | MLP, Adam, training loop | — |
| 3 | Self-pruning (importance criterion, schedule, regrowth) | — |
| 4 | Pareto sweep, cost measurement, baseline, claim | — |

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run the tests

```bash
uv run pytest
```

This runs the finite-difference gradient checks for every operation
(`tests/test_gradcheck.py`), engine behaviour tests (`tests/test_engine.py`)
and the masked-weight correctness suite (`tests/test_masking.py`).

## Layout

```
aqua/
  engine/
    tensor.py      Tensor, graph construction, backward(), all core ops
    functional.py  stable log_softmax / softmax / fused cross-entropy
    parameter.py   Parameter: trainable tensor with a read-only connectivity mask
    gradcheck.py   central-difference gradient checker
  optim/
    optimizer.py   mask-aware optimizer base (state reset on prune/revive)
    sgd.py         SGD with momentum
    adam.py        Adam / AdamW with per-entry step counts
tests/
```

## Part 1 summary

**Operations:** `+ - * /` (with full NumPy broadcasting, including reflected
operators with Python scalars and arrays), `**` with a constant exponent,
`@` (2-D and batched with broadcast batch axes), `sum`/`mean` (any axes,
`keepdims`), `reshape`, `transpose`, `exp`, `log`, `relu`, `tanh`, `sigmoid`
(overflow-free), `gelu` (tanh approximation), `log_softmax`, `softmax`, and a
fused `cross_entropy`.

**Backward:** iterative topological sort (no recursion limit), with gradients
accumulated per node, so reused sub-expressions and broadcasting both sum
correctly. Broadcast operands get their gradient reduced back to their own shape.

**Masked weights:** see [DESIGN.md](DESIGN.md#the-gradient-of-a-masked-weight).
In short, a pruned weight has `grad == 0` exactly (the true gradient of the
function computed), `dense_grad` carries the regrowth signal, and the optimizers
keep a pruned entry's value and state at exactly zero. Optimizer state is
per-entry, including Adam's step count, so a revived connection follows the
same trajectory as a freshly initialised one. `tests/test_masking.py` checks
this against exact independent reference runs.
