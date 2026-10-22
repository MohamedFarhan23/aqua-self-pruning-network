"""Part 2: layers, initialisation and the training loop, on the fast spirals task."""

import math

import numpy as np
import pytest

from aqua.data import make_spirals
from aqua.engine import Parameter, Tensor, cross_entropy, gradcheck, no_grad
from aqua.nn import MLP, Linear
from aqua.optim import SGD, Adam
from aqua.train import cosine, evaluate, train


def test_parameters_are_collected_once_in_order():
    model = MLP([4, 8, 3], seed=0)
    names = [p.name for p in model.parameters()]
    assert names == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    assert model.num_parameters() == 4 * 8 + 8 + 8 * 3 + 3
    assert [p.name for p in model.prunable()] == ["fc1.weight", "fc2.weight"]


def test_mlp_end_to_end_gradcheck():
    model = MLP([3, 5, 4, 2], activation="tanh", seed=0, dtype=np.float64)
    x = Tensor(np.random.default_rng(0).standard_normal((6, 3)))
    y = np.array([0, 1, 1, 0, 1, 0])
    params = list(model.parameters())
    gradcheck(lambda *_: cross_entropy(model(x), y), params)


def test_linear_reads_weight_through_its_mask():
    layer = Linear(3, 2, np.random.default_rng(0), dtype=np.float64)
    mask = np.array([[True, False], [False, True], [True, True]])
    layer.weight.set_mask(mask)
    layer.weight.data[~mask] = 7.0  # corrupt pruned storage
    x = np.eye(3)
    np.testing.assert_array_equal(layer(Tensor(x)).data, layer.weight.data * mask)


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_init_keeps_every_preactivation_at_unit_second_moment(activation):
    """Including the first layer (fed standardised data) and the logits."""
    model = MLP([256] * 7 + [64], activation=activation, seed=0, dtype=np.float64)
    h = Tensor(np.random.default_rng(0).standard_normal((4096, 256)))
    moments = []
    with no_grad():
        for layer in model.layers:
            h = layer(h)
            if isinstance(layer, Linear):
                moments.append(float(np.mean(h.data**2)))
    assert len(moments) == 7
    assert all(0.7 < m < 1.3 for m in moments), moments


def test_initial_loss_is_uninformed():
    data = make_spirals(seed=0)
    model = MLP([2, 64, 64, data.num_classes], seed=0)
    loss, _ = evaluate(model, data.val)
    assert abs(loss - math.log(data.num_classes)) < 0.5


@pytest.mark.parametrize("make_opt", [lambda ps: Adam(ps, lr=0.01), lambda ps: SGD(ps, lr=0.05, momentum=0.9)],
                         ids=["adam", "sgd_momentum"])
def test_learns_spirals(make_opt):
    data = make_spirals(seed=0)
    model = MLP([2, 64, 64, data.num_classes], seed=0)
    opt = make_opt(model.parameters())
    steps = 40 * math.ceil(len(data.train) / 32)
    history = train(model, opt, data, epochs=40, batch_size=32, seed=0, lr_schedule=cosine(opt.lr, steps))
    assert all(np.isfinite(h.train_loss) and np.isfinite(h.max_grad_norm) for h in history)
    assert history[-1].train_loss < 0.25 * history[0].train_loss
    _, test_acc = evaluate(model, data.test)
    assert test_acc > 0.95


def test_training_is_reproducible_bit_for_bit():
    def run():
        data = make_spirals(seed=1)
        model = MLP([2, 32, data.num_classes], seed=1)
        train(model, Adam(model.parameters(), lr=0.01), data, epochs=3, batch_size=32, seed=1)
        return [p.data.copy() for p in model.parameters()]
    for a, b in zip(run(), run()):
        np.testing.assert_array_equal(a, b)


def test_relu_propagates_nan():
    assert np.isnan(Tensor(np.array([np.nan, -1.0, 1.0])).relu().data[0])


def test_nan_loss_stops_training():
    data = make_spirals(seed=0)
    model = MLP([2, 8, data.num_classes], seed=0)
    model.linears[0].weight.data[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        train(model, Adam(model.parameters()), data, epochs=1)


def test_cosine_schedule_endpoints():
    s = cosine(1e-3, 100, final_fraction=0.1)
    assert s(0) == pytest.approx(1e-3)
    assert s(50) == pytest.approx(0.55e-3)
    assert s(100) == pytest.approx(1e-4)
    assert s(500) == pytest.approx(1e-4)


def test_parameter_dtype_follows_data():
    assert Parameter(np.zeros(3, dtype=np.float32)).dtype == np.float32
    assert Parameter(np.zeros(3)).dtype == np.float64
    assert Parameter([1, 2]).dtype == np.float64
