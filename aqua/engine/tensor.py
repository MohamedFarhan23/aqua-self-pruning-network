"""A minimal reverse-mode automatic differentiation engine over NumPy arrays.

Every differentiable operation produces a new ``Tensor`` that remembers its
parents and a ``_backward`` closure. The closure maps the upstream gradient
dL/d(out) to a tuple of gradients dL/d(parent), one per parent (``None`` when
that parent does not need one). ``Tensor.backward`` owns topological ordering
and gradient accumulation, so individual ops never mutate shared state.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Sequence

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (for evaluation and numerical gradients)."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` by summing over broadcast axes.

    Broadcasting copies an operand along (a) new leading axes and (b) axes of
    size 1. Each copy contributes to the loss, so the gradient of the original
    operand is the sum of the gradients of its copies.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


BackwardFn = Callable[[np.ndarray], Sequence["np.ndarray | None"]]


class Tensor:
    # Makes ``ndarray <op> Tensor`` dispatch to Tensor's reflected operators.
    __array_priority__ = 1000

    def __init__(self, data, requires_grad: bool = False):
        arr = np.asarray(data)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: BackwardFn | None = None
        self._op = ""

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        op = f", op={self._op}" if self._op else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{op})"

    def _lift(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return other
        # Constants adopt our dtype so float32 graphs stay float32.
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    @staticmethod
    def _make(data: np.ndarray, parents: tuple[Tensor, ...], backward: BackwardFn, op: str) -> Tensor:
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    # --------------------------------------------------------------- backward
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Accumulate dself/dleaf into ``leaf.grad`` for every leaf in the graph.

        Gradients of intermediate nodes live only in a local dict for the
        duration of the pass; only leaves (tensors created by the user, such as
        parameters) accumulate into ``.grad`` across calls, until zeroed.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("backward() without an explicit grad needs a scalar output")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.data.dtype)
        if grad.shape != self.shape:
            raise ValueError(f"grad shape {grad.shape} != tensor shape {self.shape}")

        # Iterative post-order DFS: no recursion limit on deep graphs.
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))

        grads: dict[int, np.ndarray] = {id(self): grad}
        for node in reversed(order):  # every consumer runs before its inputs
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g)):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                grads[key] = pg if key not in grads else grads[key] + pg

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other) -> Tensor:
        other = self._lift(other)
        a_shape, b_shape = self.shape, other.shape

        def backward(g):
            return _unbroadcast(g, a_shape), _unbroadcast(g, b_shape)

        return Tensor._make(self.data + other.data, (self, other), backward, "add")

    __radd__ = __add__

    def __sub__(self, other) -> Tensor:
        other = self._lift(other)
        a_shape, b_shape = self.shape, other.shape

        def backward(g):
            return _unbroadcast(g, a_shape), _unbroadcast(-g, b_shape)

        return Tensor._make(self.data - other.data, (self, other), backward, "sub")

    def __rsub__(self, other) -> Tensor:
        return self._lift(other) - self

    def __mul__(self, other) -> Tensor:
        other = self._lift(other)
        a, b = self.data, other.data

        def backward(g):
            ga = _unbroadcast(g * b, a.shape) if self.requires_grad else None
            gb = _unbroadcast(g * a, b.shape) if other.requires_grad else None
            return ga, gb

        return Tensor._make(a * b, (self, other), backward, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other) -> Tensor:
        other = self._lift(other)
        a, b = self.data, other.data

        def backward(g):
            ga = _unbroadcast(g / b, a.shape) if self.requires_grad else None
            gb = _unbroadcast(-g * a / (b * b), b.shape) if other.requires_grad else None
            return ga, gb

        return Tensor._make(a / b, (self, other), backward, "div")

    def __rtruediv__(self, other) -> Tensor:
        return self._lift(other) / self

    def __neg__(self) -> Tensor:
        return Tensor._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise TypeError("only constant exponents are supported")
        x = self.data
        p = float(exponent)

        def backward(g):
            return (g * p * x ** (p - 1),)

        return Tensor._make(x**p, (self,), backward, "pow")

    def __matmul__(self, other) -> Tensor:
        other = self._lift(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul requires operands with ndim >= 2")

        def backward(g):
            # out = a @ b  =>  dL/da = g @ b^T,  dL/db = a^T @ g  (per batch);
            # leading batch axes may broadcast, so reduce them back.
            ga = _unbroadcast(g @ np.swapaxes(b, -1, -2), a.shape) if self.requires_grad else None
            gb = _unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape) if other.requires_grad else None
            return ga, gb

        return Tensor._make(a @ b, (self, other), backward, "matmul")

    def __rmatmul__(self, other) -> Tensor:
        return self._lift(other) @ self

    # ------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        in_shape = self.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, in_shape),)

        return Tensor._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward, "sum")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # ------------------------------------------------------------------ shape
    def reshape(self, *shape) -> Tensor:
        in_shape = self.shape
        return Tensor._make(self.data.reshape(*shape), (self,), lambda g: (g.reshape(in_shape),), "reshape")

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        inverse = None if axes is None else np.argsort(axes)
        return Tensor._make(
            np.transpose(self.data, axes), (self,), lambda g: (np.transpose(g, inverse),), "transpose"
        )

    @property
    def T(self) -> Tensor:
        return self.transpose()

    # ---------------------------------------------------------- element-wise
    def exp(self) -> Tensor:
        out = np.exp(self.data)
        return Tensor._make(out, (self,), lambda g: (g * out,), "exp")

    def log(self) -> Tensor:
        x = self.data
        return Tensor._make(np.log(x), (self,), lambda g: (g / x,), "log")

    def relu(self) -> Tensor:
        # Subgradient 0 at x == 0. np.maximum propagates NaN (np.where(x > 0, x, 0)
        # would silently turn a NaN into 0 and hide a diverged weight).
        active = self.data > 0
        return Tensor._make(np.maximum(self.data, 0), (self,), lambda g: (g * active,), "relu")

    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), lambda g: (g * (1.0 - out * out),), "tanh")

    def sigmoid(self) -> Tensor:
        # Never exponentiate a positive number: exp(-|x|) is in (0, 1].
        x = self.data
        e = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(self.dtype)
        return Tensor._make(out, (self,), lambda g: (g * out * (1.0 - out),), "sigmoid")

    def gelu(self) -> Tensor:
        """GELU, tanh approximation (Hendrycks & Gimpel): exact erf needs SciPy."""
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        u = c * (x + 0.044715 * x**3)
        t = np.tanh(u)
        out = 0.5 * x * (1.0 + t)

        def backward(g):
            du = c * (1.0 + 3 * 0.044715 * x * x)
            return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du),)

        return Tensor._make(out, (self,), backward, "gelu")
