# Design notes

## The gradient of a masked weight

A pruned connection is represented by a binary mask `m` on a `Parameter` `w`.
The forward pass never reads `w` directly. It reads the effective weight
`w_eff = w ⊙ m` produced by `Parameter.masked()`. Two different quantities are
reasonably called "the gradient of a masked weight", and the engine computes
both on purpose:

| Name | Definition | On a pruned entry | Used for |
| --- | --- | --- | --- |
| `grad` | ∂L/∂w | exactly `0` | optimizer updates |
| `dense_grad` | ∂L/∂w_eff | generally ≠ 0 | regrowth / saliency |

By the chain rule, ∂L/∂w = ∂L/∂w_eff ⊙ m. `grad` is therefore the *true*
gradient of the function the network actually computes. Perturbing a pruned
weight's storage changes nothing downstream, so its derivative is 0, and the
finite-difference checker agrees (`test_masked_parameter_gradcheck`). This is
the right signal for training, because a pruned weight is not a free parameter.

`dense_grad` is the first-order answer to a different question: *if this
connection existed, at its current value of 0, which way would the loss want
it to move?* It is the regrowth signal (RigL-style) and is never fed to the
optimizer.

### Why a zero gradient is not enough

Masking the gradient is necessary but not sufficient. Momentum-based optimizers
keep moving on a zero gradient:

* **Stale momentum drift.** Adam with g = 0 still applies
  `lr · m̂ / (√v̂ + ε)` from its decaying history. A "pruned" weight keeps
  moving. It is invisible in the forward pass because of the mask, but it comes
  back with an arbitrary value on revival (`test_naive_adam_moves_a_weight_whose_gradient_is_zero`).
* **Stale moments on revival.** The revived connection inherits `m`, `v` from
  a past that no longer applies.
* **Global step count in bias correction.** Even if `m`, `v` are reset, Adam's
  bias correction `1/(1-β^t)` assumes the EMA has been accumulating for `t`
  steps. With a shared global step `T`, the first revived step becomes
  `lr · (1-β₁)/(1-β₁ᵀ) · √((1-β₂ᵀ)/(1-β₂)) · sign(g)` instead of
  `lr · sign(g)`. With default betas that is 2.51× too large at `T = 1000`, and
  the factor approaches 3.16× as `T → ∞`
  (`test_naive_global_step_overshoots_revived_entry`).

### What the optimizers do

All optimizer state is per entry, including Adam's step count `t`. On every
step (`aqua/optim/optimizer.py`):

1. Entries that are pruned now, **or were pruned at any point since the previous
   step**, have all state zeroed. `Parameter.set_mask` records 1→0 transitions,
   so a prune→revive cycle with no step in between is still caught.
2. Moments and `t` advance only on active entries.
3. Pruned entries get a zero update, and their value is forced to exactly 0.0.

Since the mask is read-only and changes only through `set_mask`, this
bookkeeping cannot be bypassed by accident.

As a result, while an entry is pruned its value and state are exactly zero. When
it is revived it starts at 0, so the network's function is continuous across
the revival, and it follows exactly the trajectory of a freshly initialised
optimizer on that coordinate. Other coordinates are unaffected bit for bit.
`test_prune_and_revive_matches_independent_references` checks all three
properties on a separable loss, where exact reference trajectories exist, for
Adam, AdamW and SGD with momentum.

## Weight initialisation

For `z = h W` with i.i.d. `W_ij ~ N(0, s²)`, `E[z²] = fan_in · s² · E[h²]`. The
init keeps `E[z²] = 1` at every layer by setting `s² = gain / fan_in`, where
the gain undoes whatever produced the layer's **input**:

| input to the layer | E[h²] / E[z²] of previous layer | gain |
| --- | --- | --- |
| standardised data (first layer) | 1 | 1 |
| ReLU | 1/2 | 2 (He) |
| GELU (tanh approx.) | 0.4252 (numerical integration) | 2.35 |
| tanh | ≈ 1 near 0 | 1 (LeCun) |

The output layer consumes ReLU outputs, so it also uses gain 2 and the logits
start at unit variance. Expanding log-sum-exp to second order gives an initial
loss ≈ `ln C + σ²(C−1)/(2C)`. With the measured logit variance σ² ≈ 0.8–0.9,
that predicts ~2.65 against an observed 2.60, close to the uninformed
ln 10 = 2.30 and never confidently wrong. `test_init_keeps_every_preactivation_at_unit_second_moment`
checks that a 7-layer ReLU and a 7-layer GELU network keep every
pre-activation, logits included, within [0.7, 1.3].

My first version keyed the gain on the activation *after* the layer. That gives
the first layer gain 2 on standardised data and the output layer gain 1. For a
bias-free ReLU network the two errors cancel exactly at init (ReLU is
positively homogeneous: scaling fc1 by 1/√2 and fc3 by √2 leaves the function
unchanged, and the epoch-0 loss was identical to 4 decimals). They still put
the first hidden layer at E[z²] = 2, which changes how the gradient is
distributed across layers. The test caught it.

## Importance criterion

*Part 3.*

## Engine bottlenecks

Profile of one LeNet-300-100 training step (batch 128, float32, 16-core
OpenBLAS), measured while building Part 2:

| phase | first version | now |
| --- | --- | --- |
| forward | 1.4 ms | 1.0 ms |
| backward | 0.8 ms | 0.7 ms |
| global grad norm (NaN guard) | 0.6 ms | 0.6 ms |
| optimizer step (Adam) | **23.2 ms** | **2.5 ms** |

The autodiff itself is cheap because the graph is ~10 nodes of large matmuls.
The cost was in the optimizer's per-entry bookkeeping: an element-wise
`β ** t` in float64 for bias correction, plus `np.where` and boolean-index
passes even on dense parameters. That now uses a lookup table for `1 − βᵗ`
gathered by `t`, in-place float32 updates, and a mask multiply only for
masked parameters. That cut epoch time from ~9 s to ~2 s. The mutation checks
on the masking suite were rerun after the rewrite.

Remaining bottlenecks, roughly in order:

1. **Optimizer memory traffic.** Adam still makes ~10 full passes over every
   parameter per step (m, v, the t gather, the divisions). A fused kernel
   (Numba/C) would do one pass.
2. **Temporaries in the engine.** Every op allocates its output and every
   backward allocates its gradient. Intermediate gradients are freed after use
   (they live in a dict local to `backward()`), but there are no in-place or
   buffer-reusing kernels.
3. **Python dispatch per op.** This is negligible for a 3-layer MLP. It would
   dominate for graphs with many small ops, where tracing and fusing would help.
4. **Dense matmul on sparse weights.** Training multiplies by the mask, so a
   pruned network trains no faster. Part 4 measures inference with a separate
   sparse forward path.

## Serving a self-pruned model

*To be completed.*
