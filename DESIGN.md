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

## Importance criterion

*Part 3.*

## Engine bottlenecks

*To be completed with profiling data from Part 2.*

## Serving a self-pruned model

*To be completed.*
