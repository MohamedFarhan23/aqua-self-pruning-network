"""Part 1 correctness requirement: masked weights in the engine and the optimizers.

Most tests use a separable loss, L(w) = sum_i (w_i * m_i - c_i)^2. Each
coordinate's gradient depends only on that coordinate, and both optimizers are
element-wise, so every coordinate follows its own trajectory. That gives exact
reference answers:

* pruning coordinate j must leave every other coordinate's trajectory
  bit-for-bit unchanged;
* after revival, coordinate j must follow exactly the trajectory of a fresh
  optimizer started on that coordinate alone from w = 0.
"""

import numpy as np
import pytest

from aqua.engine import Parameter, Tensor
from aqua.optim import SGD, Adam

N = 6
J = 2  # the coordinate we prune and revive
PRUNE_AT, REVIVE_AT, TOTAL = 10, 25, 60
TARGET = np.linspace(-1.5, 2.0, N)


def separable_loss(w: Parameter, target=TARGET) -> Tensor:
    return ((w.masked() - target) ** 2).sum()


def run(make_opt, w0, schedule=None, steps=TOTAL, target=TARGET):
    """Train on the separable loss; ``schedule`` maps step -> mask installed before that step."""
    w = Parameter(w0.copy())
    opt = make_opt([w])
    traj = []
    for step in range(steps):
        if schedule and step in schedule:
            w.set_mask(schedule[step])
        opt.zero_grad()
        separable_loss(w, target).backward()
        opt.step()
        traj.append(w.data.copy())
    return w, opt, np.array(traj)


def prune_revive_schedule():
    pruned = np.ones(N, dtype=bool)
    pruned[J] = False
    return {PRUNE_AT: pruned, REVIVE_AT: np.ones(N, dtype=bool)}


OPTIMIZERS = {
    "adam": lambda ps: Adam(ps, lr=0.05),
    "adamw": lambda ps: Adam(ps, lr=0.05, weight_decay=0.1),
    "sgd_momentum": lambda ps: SGD(ps, lr=0.02, momentum=0.9),
}


# --------------------------------------------------------------------- engine
def test_set_mask_zeroes_storage_and_mask_is_read_only():
    w = Parameter(np.ones((3, 3)))
    mask = np.eye(3, dtype=bool)
    w.set_mask(mask)
    np.testing.assert_array_equal(w.data, np.eye(3))
    with pytest.raises(ValueError):
        w.mask[0, 1] = True  # bypassing set_mask would skip the optimizer reset bookkeeping
    mask[0, 1] = True  # the caller's array is copied, not aliased
    assert not w.mask[0, 1]


def test_masked_forward_equals_physically_removed_connections():
    rng = np.random.default_rng(1)
    w = Parameter(rng.standard_normal((5, 4)))
    mask = np.ones((5, 4), dtype=bool)
    mask[1, :] = False  # input feature 1 fully disconnected
    w.set_mask(mask)
    w.data[1, :] = 123.0  # storage corruption must not leak into the forward pass
    x = rng.standard_normal((3, 5))
    out = (Tensor(x) @ w.masked()).data
    reduced = np.delete(x, 1, axis=1) @ np.delete(w.data, 1, axis=0)
    np.testing.assert_allclose(out, reduced, rtol=1e-12)


def test_grad_of_masked_weight_is_exact_zero_and_dense_grad_is_regrowth_signal():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 5))
    y = rng.standard_normal((8, 4))
    w = Parameter(rng.standard_normal((5, 4)))
    mask = rng.random((5, 4)) > 0.4
    w.set_mask(mask)

    loss = ((Tensor(x) @ w.masked() - y) ** 2).mean() * 1e6  # large upstream gradient
    loss.backward()

    assert np.all(w.grad[~mask] == 0.0)  # exactly, not approximately
    np.testing.assert_array_equal(w.grad[mask], w.dense_grad[mask])
    assert np.all(w.dense_grad[~mask] != 0.0)

    # dense_grad on a pruned entry = d loss / d(effective weight): what reviving
    # the connection at value 0 and nudging it would do to the loss.
    w_eff = w.data * mask
    def loss_of(eff):
        return float(((x @ eff - y) ** 2).mean() * 1e6)
    eps = 1e-6
    for i, k in zip(*np.nonzero(~mask)):
        plus, minus = w_eff.copy(), w_eff.copy()
        plus[i, k] += eps
        minus[i, k] -= eps
        numeric = (loss_of(plus) - loss_of(minus)) / (2 * eps)
        assert w.dense_grad[i, k] == pytest.approx(numeric, rel=1e-5)


# ------------------------------------------------------------------ optimizers
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_pruned_weight_stays_exactly_zero_with_zero_state(name):
    w0 = np.random.default_rng(3).standard_normal(N)
    w, opt, traj = run(OPTIMIZERS[name], w0, {PRUNE_AT: prune_revive_schedule()[PRUNE_AT]})
    assert np.all(traj[PRUNE_AT:, J] == 0.0)
    for buf in opt.state_for(w).values():
        assert buf[J] == 0


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizer_ignores_gradient_leaking_onto_pruned_entries(name):
    """Defence in depth: a layer that forgets .masked() hands the optimizer a
    non-zero gradient on pruned entries. The mask still wins."""
    w = Parameter(np.random.default_rng(6).standard_normal(N))
    opt = OPTIMIZERS[name]([w])
    w.set_mask(prune_revive_schedule()[PRUNE_AT])
    for _ in range(5):
        opt.zero_grad()
        ((w - TARGET) ** 2).sum().backward()  # bypasses the mask
        assert w.grad[J] != 0.0
        opt.step()
        assert w.data[J] == 0.0
        for buf in opt.state_for(w).values():
            assert buf[J] == 0


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_prune_and_revive_matches_independent_references(name):
    make_opt = OPTIMIZERS[name]
    w0 = np.random.default_rng(4).standard_normal(N)

    _, _, uninterrupted = run(make_opt, w0)
    _, _, traj = run(make_opt, w0, prune_revive_schedule())

    others = np.arange(N) != J
    # 1. Pruning and reviving J does not perturb any other coordinate, bit for bit.
    np.testing.assert_array_equal(traj[:, others], uninterrupted[:, others])
    # 2. Before pruning, J is untouched.
    np.testing.assert_array_equal(traj[:PRUNE_AT, J], uninterrupted[:PRUNE_AT, J])
    # 3. While pruned, J is exactly 0.
    assert np.all(traj[PRUNE_AT:REVIVE_AT, J] == 0.0)
    # 4. After revival, J is a fresh optimizer started from 0 on its own target.
    _, _, fresh = run(make_opt, np.zeros(1), steps=TOTAL - REVIVE_AT, target=TARGET[J:J + 1])
    np.testing.assert_allclose(traj[REVIVE_AT:, J], fresh[:, 0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_prune_and_revive_between_two_steps_still_resets_state(name):
    """Mask goes 1 -> 0 -> 1 with no optimizer step in between: set_mask has
    zeroed the weight, so its old momentum must not survive."""
    make_opt = OPTIMIZERS[name]
    w0 = np.random.default_rng(5).standard_normal(N)
    w, opt, _ = run(make_opt, w0, steps=PRUNE_AT)

    pruned = np.ones(N, dtype=bool)
    pruned[J] = False
    w.set_mask(pruned)
    w.set_mask(np.ones(N, dtype=bool))
    traj = []
    for _ in range(TOTAL - PRUNE_AT):
        opt.zero_grad()
        separable_loss(w).backward()
        opt.step()
        traj.append(w.data[J])

    _, _, fresh = run(make_opt, np.zeros(1), steps=TOTAL - PRUNE_AT, target=TARGET[J:J + 1])
    np.testing.assert_allclose(traj, fresh[:, 0], rtol=0, atol=1e-15)


def test_revived_adam_entry_restarts_bias_correction():
    """First Adam step after revival has magnitude ~lr (fresh t = 1), not
    lr * (1 - b1) / sqrt(1 - b2) ~= 3.16 lr as a shared global step count gives."""
    lr = 0.01
    w = Parameter(np.full(2, 1.0))
    opt = Adam([w], lr=lr)
    for _ in range(500):
        opt.zero_grad()
        separable_loss(w, np.array([5.0, 5.0])).backward()
        opt.step()
    w.set_mask(np.array([True, False]))
    for _ in range(5):
        opt.zero_grad()
        separable_loss(w, np.array([5.0, 5.0])).backward()
        opt.step()
    w.set_mask(np.array([True, True]))
    opt.zero_grad()
    separable_loss(w, np.array([5.0, 5.0])).backward()
    opt.step()
    assert opt.state_for(w)["t"][1] == 1
    assert w.data[1] == pytest.approx(lr, rel=1e-6)


# ------------------------------------------- the bugs those tests guard against
def naive_adam_step(w, g, m, v, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    """Textbook Adam: one global t, no knowledge of the mask."""
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g * g
    w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)


def test_naive_adam_moves_a_weight_whose_gradient_is_zero():
    """Shows the bug exists: a zero (masked) gradient does not freeze Adam."""
    w = np.array([1.0])
    m, v = np.zeros(1), np.zeros(1)
    for t in range(1, 11):
        naive_adam_step(w, np.array([2.0 * (w[0] - 3.0)]), m, v, t)
    w[:] = 0.0  # "pruned"
    for t in range(11, 21):
        naive_adam_step(w, np.zeros(1), m, v, t)  # engine correctly reports grad = 0
    assert abs(w[0]) > 0.1  # yet it drifted far from zero on stale momentum


@pytest.mark.parametrize("global_t", [1000, 10_000])
def test_naive_global_step_overshoots_revived_entry(global_t):
    lr, b1, b2 = 0.05, 0.9, 0.999
    w = np.array([0.0])
    m, v = np.zeros(1), np.zeros(1)  # moments correctly reset on revival...
    naive_adam_step(w, np.array([-1.0]), m, v, t=global_t)  # ...but t is the global step
    # A fresh entry (t = 1) would move by exactly lr. With global t = T it moves by
    # lr * (1-b1)/(1-b1^T) * sqrt((1-b2^T)/(1-b2)): 2.51x at T=1000, -> 3.16x as T -> inf.
    factor = (1 - b1) / (1 - b1**global_t) * np.sqrt((1 - b2**global_t) / (1 - b2))
    assert w[0] == pytest.approx(lr * factor, rel=1e-6)
    assert factor > 2.5
