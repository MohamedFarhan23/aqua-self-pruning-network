"""Finite-difference gradient checks for every differentiable operation."""

import numpy as np
import pytest

from aqua.engine import Parameter, Tensor, cross_entropy, gradcheck, log_softmax, softmax

rng = np.random.default_rng(0)


def t(*shape, low=None, high=None):
    """float64 leaf requiring grad. With ``low``, values are drawn with |x| in [low, high]."""
    if low is None:
        data = rng.standard_normal(shape)
    else:
        data = rng.uniform(low, high, shape) * rng.choice([-1.0, 1.0], shape)
    return Tensor(data, requires_grad=True)


def pos(*shape):
    return Tensor(rng.uniform(0.5, 2.0, shape), requires_grad=True)


# (id, fn, inputs-factory). Factories keep each case's data independent.
CASES = [
    # element-wise arithmetic, including every broadcasting pattern
    ("add", lambda a, b: a + b, lambda: [t(3, 4), t(3, 4)]),
    ("add_bcast_ND_D", lambda a, b: a + b, lambda: [t(5, 4), t(4)]),
    ("add_bcast_D_ND", lambda a, b: a + b, lambda: [t(4), t(5, 4)]),
    ("add_bcast_N1_1D", lambda a, b: a + b, lambda: [t(5, 1), t(1, 4)]),
    ("add_bcast_scalar_tensor", lambda a, b: a + b, lambda: [t(3, 4), t()]),
    ("add_python_scalar", lambda a: a + 2.5, lambda: [t(3, 4)]),
    ("radd_python_scalar", lambda a: 2.5 + a, lambda: [t(3, 4)]),
    ("sub_bcast", lambda a, b: a - b, lambda: [t(5, 4), t(4)]),
    ("rsub_scalar", lambda a: 1.0 - a, lambda: [t(3, 4)]),
    ("mul", lambda a, b: a * b, lambda: [t(3, 4), t(3, 4)]),
    ("mul_bcast_3d", lambda a, b: a * b, lambda: [t(4, 1, 3), t(2, 3)]),
    ("mul_const_array", lambda a: a * np.arange(4.0), lambda: [t(3, 4)]),
    ("div", lambda a, b: a / b, lambda: [t(3, 4), t(3, 4, low=0.5, high=2.0)]),
    ("div_bcast", lambda a, b: a / b, lambda: [t(5, 4), t(4, low=0.5, high=2.0)]),
    ("rdiv_scalar", lambda a: 3.0 / a, lambda: [t(3, 4, low=0.5, high=2.0)]),
    ("neg", lambda a: -a, lambda: [t(3, 4)]),
    ("pow_2", lambda a: a**2, lambda: [t(3, 4)]),
    ("pow_half", lambda a: a**0.5, lambda: [pos(3, 4)]),
    ("pow_neg", lambda a: a**-1.5, lambda: [pos(3, 4)]),
    # matrix multiplication
    ("matmul", lambda a, b: a @ b, lambda: [t(5, 3), t(3, 4)]),
    ("matmul_batched", lambda a, b: a @ b, lambda: [t(2, 5, 3), t(2, 3, 4)]),
    ("matmul_batch_bcast", lambda a, b: a @ b, lambda: [t(2, 5, 3), t(3, 4)]),
    ("linear", lambda x, w, b: x @ w + b, lambda: [t(6, 3), t(3, 4), t(4)]),
    # reductions
    ("sum_all", lambda a: a.sum(), lambda: [t(3, 4)]),
    ("sum_axis0", lambda a: a.sum(axis=0), lambda: [t(3, 4)]),
    ("sum_axis1_keepdims", lambda a: a.sum(axis=1, keepdims=True), lambda: [t(3, 4)]),
    ("sum_axis_tuple", lambda a: a.sum(axis=(0, 2)), lambda: [t(2, 3, 4)]),
    ("sum_neg_axis", lambda a: a.sum(axis=-1), lambda: [t(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), lambda: [t(3, 4)]),
    ("mean_axis0", lambda a: a.mean(axis=0), lambda: [t(3, 4)]),
    ("mean_axis_tuple_keepdims", lambda a: a.mean(axis=(1, 2), keepdims=True), lambda: [t(2, 3, 4)]),
    # shape
    ("reshape", lambda a: a.reshape(4, 3) * np.arange(12.0).reshape(4, 3), lambda: [t(3, 4)]),
    ("transpose", lambda a: a.T @ a, lambda: [t(3, 4)]),
    ("transpose_axes", lambda a: a.transpose((2, 0, 1)), lambda: [t(2, 3, 4)]),
    # non-linearities (relu inputs kept away from the kink at 0)
    ("relu", lambda a: a.relu(), lambda: [t(4, 5, low=0.1, high=2.0)]),
    ("tanh", lambda a: a.tanh(), lambda: [t(4, 5)]),
    ("sigmoid", lambda a: a.sigmoid(), lambda: [t(4, 5)]),
    ("sigmoid_saturated", lambda a: a.sigmoid(), lambda: [t(4, 5, low=10.0, high=30.0)]),
    ("gelu", lambda a: a.gelu(), lambda: [t(4, 5)]),
    ("exp", lambda a: a.exp(), lambda: [t(4, 5)]),
    ("log", lambda a: a.log(), lambda: [pos(4, 5)]),
    # softmax family
    ("log_softmax", lambda a: log_softmax(a, axis=-1), lambda: [t(4, 5)]),
    ("log_softmax_axis0", lambda a: log_softmax(a, axis=0), lambda: [t(4, 5)]),
    ("softmax", lambda a: softmax(a), lambda: [t(4, 5)]),
    ("cross_entropy", lambda a: cross_entropy(a, np.array([0, 3, 1, 4])), lambda: [t(4, 5)]),
    ("cross_entropy_large_logits", lambda a: cross_entropy(a * 50.0, np.array([0, 3, 1, 4])),
     lambda: [t(4, 5)]),
    # graph structure: reuse and fan-in must accumulate, not overwrite
    ("reuse_square", lambda a: a * a + a, lambda: [t(3, 4)]),
    ("diamond", lambda a: (a.tanh() * a.sigmoid()) + a.exp(), lambda: [t(3, 4)]),
    ("mlp", lambda x, w1, b1, w2, b2: cross_entropy((x @ w1 + b1).tanh() @ w2 + b2, np.array([0, 2, 1, 2, 0])),
     lambda: [Tensor(rng.standard_normal((5, 4))), t(4, 6), t(6), t(6, 3), t(3)]),
]


@pytest.mark.parametrize("name,fn,make_inputs", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(name, fn, make_inputs):
    gradcheck(fn, make_inputs())


def test_masked_parameter_gradcheck():
    """Finite differences of the *masked* function agree with .grad, including
    the zeros on pruned entries: perturbing a pruned weight changes nothing."""
    w = Parameter(rng.standard_normal((4, 3)))
    w.set_mask(rng.random((4, 3)) > 0.5)
    w.data[:] = rng.standard_normal((4, 3))  # even if pruned storage were non-zero
    x = Tensor(rng.standard_normal((5, 4)))
    gradcheck(lambda w: (x @ w.masked()).tanh(), [w])
    assert np.all(w.grad[~w.mask] == 0.0)
