"""Part 3: criteria, schedule, the pruning controller and regrowth."""

import math

import numpy as np
import pytest

from aqua.data import make_spirals
from aqua.engine import Tensor, cross_entropy
from aqua.nn import MLP
from aqua.optim import Adam
from aqua.prune import CRITERIA, CubicSchedule, Pruner, Stats, per_example_fisher
from aqua.prune.pruner import batch_stats
from aqua.train import cosine, evaluate, train


# ----------------------------------------------------------------- schedule
def test_cubic_schedule_shape():
    s = CubicSchedule(final_sparsity=0.9, start_step=100, end_step=1100, interval=100)
    assert s.sparsity(0) == 0.0 and s.sparsity(100) == 0.0
    assert s.sparsity(1100) == pytest.approx(0.9) and s.sparsity(5000) == pytest.approx(0.9)
    assert s.sparsity(600) == pytest.approx(0.9 * (1 - 0.5**3))
    ramp = [s.sparsity(t) for t in range(100, 1101, 100)]
    increments = np.diff(ramp)
    assert np.all(increments > 0) and np.all(np.diff(increments) < 0)  # monotone, decelerating
    events = [t for t in range(0, 2000) if s.is_event(t)]
    assert events == list(range(100, 1101, 100))


# ------------------------------------------------------- per-example Fisher
def test_per_example_fisher_matches_explicit_per_example_gradients():
    rng = np.random.default_rng(0)
    model = MLP([4, 6, 3], seed=0, dtype=np.float64)
    for layer in model.linears:
        layer.capture = True
    x, y = rng.standard_normal((7, 4)), rng.integers(0, 3, 7)
    cross_entropy(model(Tensor(x)), y).backward()
    fast = [per_example_fisher(layer.input, layer.output_grad) for layer in model.linears]

    explicit = [np.zeros_like(layer.weight.data) for layer in model.linears]
    for n in range(len(y)):  # loss of example n alone = per-example loss l_n
        for p in model.parameters():
            p.zero_grad()
        cross_entropy(model(Tensor(x[n:n + 1])), y[n:n + 1]).backward()
        for acc, layer in zip(explicit, model.linears):
            acc += layer.weight.grad**2 / len(y)
    for a, b in zip(fast, explicit):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


def test_batch_stats_leave_no_gradients_and_restore_capture():
    data = make_spirals(seed=0)
    model = MLP([2, 16, data.num_classes], seed=0)
    stats = batch_stats(model, data.train.x[:300], data.train.y[:300], chunk=128)
    assert len(stats) == 2 and all(st.fisher.min() >= 0 for st in stats)
    assert all(p.grad is None for p in model.parameters())
    assert not any(layer.capture for layer in model.linears)


def test_taylor2_predicts_loss_change_of_removing_a_weight():
    """On a model whose loss is exactly quadratic in w (linear regression), with
    exact g and with H in place of F, (*) is exact. The test checks the sign
    conventions and the 1/2 factor."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    w = rng.standard_normal(5)

    def loss(w):
        return 0.5 * np.mean((x @ w - y) ** 2)

    g = x.T @ (x @ w - y) / len(y)
    h_diag = np.mean(x * x, axis=0)
    predicted = CRITERIA["taylor2"](w, Stats(grad=g, fisher=h_diag))
    for i in range(5):
        w_removed = w.copy()
        w_removed[i] = 0.0
        assert predicted[i] == pytest.approx(loss(w_removed) - loss(w), rel=1e-10)


def test_kfac_obs_matches_explicit_inverse_of_kronecker_fisher():
    rng = np.random.default_rng(2)
    xa, ug = rng.standard_normal((40, 4)), rng.standard_normal((40, 3))
    a, g = xa.T @ xa / 40, ug.T @ ug / 40
    w = rng.standard_normal((4, 3))
    lam = 0.1
    damped = lambda m: m + lam * np.trace(m) / len(m) * np.eye(len(m))
    # Row-major vec(W): index (i, j) -> i * 3 + j, matching kron(A, G).
    f_inv = np.linalg.inv(np.kron(damped(a), damped(g)))
    expected = w.ravel() ** 2 / (2 * np.diag(f_inv))
    got = CRITERIA["obs"](w, Stats(np.zeros_like(w), np.zeros_like(w), a, g), damping=lam)
    np.testing.assert_allclose(got.ravel(), expected, rtol=1e-10)


def test_obs_compensation_lowers_cost_for_correlated_inputs():
    """Two perfectly redundant inputs: OBS knows either weight can be removed
    almost for free (the twin compensates); OBD does not."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((500, 1))
    x = np.hstack([base, base + 1e-3 * rng.standard_normal((500, 1)), rng.standard_normal((500, 1))])
    a = x.T @ x / 500
    g = np.ones((1, 1))
    w = np.ones((3, 1))
    st = Stats(np.zeros_like(w), np.diag(a)[:, None] * g[0, 0], a, g)
    obs_cost = CRITERIA["obs"](w, st, damping=1e-6).ravel()
    obd_cost = CRITERIA["obd"](w, st).ravel()
    assert obs_cost[0] < 0.01 * obs_cost[2] and obs_cost[1] < 0.01 * obs_cost[2]
    assert obd_cost[0] > 0.5 * obd_cost[2]


def test_obs_requires_calibration_batches():
    data, model, opt, pruner, total = _setup("obs")
    with pytest.raises(ValueError, match="calibration"):
        train(model, opt, data, epochs=5, batch_size=32, seed=0, on_step=pruner.on_step)


# ------------------------------------------------------------------- pruner
def test_taylor2_collapses_layers_when_pruning_sets():
    """Documents the failure that ruled taylor2 out (DESIGN.md): with accurate
    gradients its signed first-order term removes whole coherent blocks."""
    data, model, opt, pruner, total = _setup("taylor2", calibrate=True)
    train(model, opt, data, epochs=30, batch_size=32, seed=0, lr_schedule=cosine(0.01, total),
          on_step=pruner.on_step)
    assert min(pruner.layer_density().values()) < 0.05
    _, acc = evaluate(model, data.test)
    assert acc < 0.8


def _setup(criterion="obs", regrow=False, sparsity=0.8, epochs=30, seed=0, calibrate=False):
    data = make_spirals(seed=seed)
    model = MLP([2, 64, 64, data.num_classes], seed=seed)
    steps_per_epoch = math.ceil(len(data.train) / 32)
    total = epochs * steps_per_epoch
    schedule = CubicSchedule(sparsity, start_step=steps_per_epoch * 3, end_step=int(total * 0.6), interval=20)
    calibration = (data.train.x, data.train.y) if calibrate else None
    pruner = Pruner(model, schedule, criterion=criterion, regrow=regrow, regrow_fraction=0.2,
                    calibration=calibration, calibration_size=256)
    opt = Adam(model.parameters(), lr=0.01)
    return data, model, opt, pruner, total


@pytest.mark.parametrize("calibrate", [False, True], ids=["ema", "batch"])
@pytest.mark.parametrize("criterion", sorted(c for c in CRITERIA if c != "taylor2"))
def test_pruner_hits_exact_budget_with_real_zeros(criterion, calibrate):
    if criterion == "obs" and not calibrate:
        pytest.skip("obs needs calibration batches")
    data, model, opt, pruner, total = _setup(criterion, calibrate=calibrate)
    train(model, opt, data, epochs=30, batch_size=32, seed=0, lr_schedule=cosine(0.01, total),
          on_step=pruner.on_step)
    k = round(0.2 * pruner.total)
    assert pruner.num_active == k
    for p in model.prunable():
        assert np.count_nonzero(p.data) <= p.num_active
        assert np.all(p.data[~p.mask] == 0.0)
    _, acc = evaluate(model, data.test)
    assert acc > 0.9, f"{criterion}: {acc}"


def test_without_regrowth_masks_only_shrink():
    data, model, opt, pruner, total = _setup("obd")
    snapshots = []
    pruner_on_step = pruner.on_step

    def on_step(step):
        pruner_on_step(step)
        if pruner.schedule.is_event(step):
            snapshots.append([p.mask.copy() for p in model.prunable()])

    train(model, opt, data, epochs=30, batch_size=32, seed=0, on_step=on_step)
    for before, after in zip(snapshots, snapshots[1:]):
        for b, a in zip(before, after):
            assert not np.any(a & ~b)
    assert all(e.revived == 0 for e in pruner.events)


@pytest.mark.parametrize("calibrate", [False, True], ids=["ema", "batch"])
def test_regrowth_revives_connections_safely(calibrate):
    data, model, opt, pruner, total = _setup("obs" if calibrate else "obd", regrow=True, calibrate=calibrate)
    observed = {"checked": 0}
    pruner_on_step = pruner.on_step

    def on_step(step):
        before = [p.mask.copy() for p in model.prunable()]
        pruner_on_step(step)
        for p, b in zip(model.prunable(), before):
            revived = p.mask & ~b
            if revived.any():
                # revived connections start at exactly 0 ...
                assert np.all(p.data[revived] == 0.0)
                observed["checked"] += 1

    train(model, opt, data, epochs=30, batch_size=32, seed=0, lr_schedule=cosine(0.01, total), on_step=on_step)
    assert observed["checked"] > 0, "no regrowth happened; the test would be vacuous"
    assert sum(e.revived for e in pruner.events) > 0
    # ... and the cap decays to 0, so the last event revives nothing
    assert pruner.events[-1].revived == 0
    assert pruner.num_active == round(0.2 * pruner.total)


def test_regrowth_respects_cap_and_skips_zero_gain_connections():
    model = MLP([4, 6, 2], seed=0, dtype=np.float64)
    schedule = CubicSchedule(0.5, start_step=1, end_step=1000, interval=1)
    pruner = Pruner(model, schedule, criterion="taylor2", regrow=True, regrow_fraction=1.0)
    w = model.linears[0].weight
    mask = np.ones(w.shape, dtype=bool)
    mask[:, :3] = False  # 12 pruned
    w.set_mask(mask)
    for p in pruner.params:
        p.set_mask(p.mask)
    zeros = [np.zeros(p.shape) for p in pruner.params]
    pruner.stats = [Stats(z.copy(), z + 1.0) for z in zeros]
    pruner.stats[0].grad[0, :3] = [3.0, 2.0, 1.0]  # only 3 pruned entries have non-zero gain
    values = [np.zeros(p.shape) for p in pruner.params]
    pruner._add_regrowth_candidates(values, k=1000, step=1)  # cap far above 3
    assert np.count_nonzero(values[0][~mask]) == 3


def test_revived_connection_gets_fresh_optimizer_state_and_grace():
    model = MLP([3, 4, 2], seed=0)
    schedule = CubicSchedule(0.5, start_step=1, end_step=100, interval=1)
    pruner = Pruner(model, schedule, criterion="magnitude", grace=5)
    opt = Adam(model.parameters(), lr=0.01)
    w = model.linears[0].weight
    x, y = np.random.default_rng(0).standard_normal((8, 3)), np.array([0, 1] * 4)
    for _ in range(3):
        opt.zero_grad()
        cross_entropy(model(Tensor(x)), y).backward()
        opt.step()

    mask = w.mask.copy()
    mask[0, 0] = False
    w.set_mask(mask)
    opt.zero_grad()
    cross_entropy(model(Tensor(x)), y).backward()
    opt.step()
    mask[0, 0] = True
    w.set_mask(mask)  # revive
    pruner.revived_at[0][0, 0] = 10
    opt.zero_grad()
    cross_entropy(model(Tensor(x)), y).backward()
    opt.step()
    assert opt.state_for(w)["t"][0, 0] == 1
    assert abs(w.data[0, 0]) == pytest.approx(0.01, rel=1e-3)  # first fresh Adam step ~ lr
    pruner._prune(12)  # 2 steps after revival < grace 5: protected despite tiny magnitude
    assert w.mask[0, 0]
