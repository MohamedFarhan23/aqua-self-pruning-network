"""Engine behaviour that finite differences alone do not cover."""

import numpy as np
import pytest

from aqua.engine import Tensor, cross_entropy, no_grad


def test_broadcast_add_grad_is_summed_over_batch():
    x = Tensor(np.ones((5, 3)), requires_grad=True)
    b = Tensor(np.zeros(3), requires_grad=True)
    (x + b).sum().backward()
    np.testing.assert_array_equal(b.grad, np.full(3, 5.0))
    np.testing.assert_array_equal(x.grad, np.ones((5, 3)))


def test_leaf_grads_accumulate_across_backward_calls():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 3.0).sum().backward()
    (x * 3.0).sum().backward()
    np.testing.assert_array_equal(x.grad, [6.0, 6.0])
    x.zero_grad()
    (x * 3.0).sum().backward()
    np.testing.assert_array_equal(x.grad, [3.0, 3.0])


def test_shared_subexpression_visited_once():
    # y is consumed by two branches; its backward must run once with the summed grad.
    x = Tensor(np.array(2.0), requires_grad=True)
    y = x * x
    (y + y * 3.0).backward()  # d/dx 4x^2 = 8x
    assert x.grad == pytest.approx(16.0)


def test_deep_graph_no_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(20_000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_non_scalar_backward_requires_grad_argument():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()
    (x * 2.0).backward(np.array([1.0, 0.0, -1.0]))
    np.testing.assert_array_equal(x.grad, [2.0, 0.0, -2.0])


def test_no_grad_builds_no_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2.0).sum()
    assert not y.requires_grad and y._parents == ()


def test_constants_receive_no_grad():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    c = Tensor(np.ones((2, 2)))
    (x @ c).sum().backward()
    assert c.grad is None and x.grad is not None


def test_float32_is_preserved():
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    y = ((x * 2.0 + 1.0) @ np.ones((3, 2), dtype=np.float32)).tanh().mean()
    y.backward()
    assert y.dtype == np.float32 and x.grad.dtype == np.float32


def test_cross_entropy_is_stable_for_huge_logits():
    logits = Tensor(np.array([[1e4, 0.0, -1e4], [0.0, 1e4, 0.0]]), requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 2]))
    loss.backward()
    # Sample 0 is predicted perfectly (loss 0); sample 1 puts all mass on the wrong class (loss 1e4).
    assert loss.item() == pytest.approx(1e4 / 2)
    assert np.all(np.isfinite(logits.grad))
    np.testing.assert_allclose(logits.grad, [[0.0, 0.0, 0.0], [0.0, 0.5, -0.5]], atol=1e-12)


def test_sigmoid_is_stable_at_extremes():
    x = Tensor(np.array([-1000.0, 0.0, 1000.0]), requires_grad=True)
    y = x.sigmoid()
    y.sum().backward()
    np.testing.assert_array_equal(y.data, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(x.grad))
