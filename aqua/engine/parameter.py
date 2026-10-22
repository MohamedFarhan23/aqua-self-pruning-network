"""Trainable tensors with a binary connectivity mask.

Invariants maintained for every pruned entry (mask == False):

1. ``data`` is exactly 0.0 (enforced by ``set_mask`` and by the optimizers).
2. The forward pass sees ``data * mask``, so the entry contributes an exact
   zero regardless of what ``data`` holds.
3. ``grad`` (dL/dw of the function actually computed) is exactly 0.0, because
   w enters the graph only through w * m and m = 0.
4. ``dense_grad`` holds dL/d(w*m): the gradient the connection *would* receive
   if it were present at its current value of zero. This is the regrowth
   signal; it is never fed to the optimizer.

Optimizers reset per-entry state for any entry reported by ``consume_dropped``
(entries that went from active to pruned since the last optimizer step), so a
prune-then-revive cycle always restarts that connection from fresh state.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, name: str = ""):
        super().__init__(np.array(data, dtype=np.result_type(np.asarray(data).dtype, np.float32)),
                         requires_grad=True)
        self.name = name
        self._mask: np.ndarray | None = None
        self._dense_grad: np.ndarray | None = None
        self._dropped: np.ndarray | None = None

    # ------------------------------------------------------------------- mask
    @property
    def mask(self) -> np.ndarray | None:
        """Read-only boolean mask (True = active), or None for a dense parameter."""
        return self._mask

    def set_mask(self, mask: np.ndarray) -> None:
        """Install a new connectivity mask; pruned entries are zeroed immediately.

        Revived entries keep whatever ``data`` holds, which by invariant 1 is 0.
        A caller that wants a different revival value writes it after this call.
        """
        new = np.array(mask, dtype=bool)
        if new.shape != self.shape:
            raise ValueError(f"mask shape {new.shape} != parameter shape {self.shape}")
        old = self._mask if self._mask is not None else np.ones(self.shape, dtype=bool)
        dropped = old & ~new
        self._dropped = dropped if self._dropped is None else self._dropped | dropped
        new.flags.writeable = False  # all mask changes must go through set_mask
        self._mask = new
        self.data[~new] = 0.0

    def consume_dropped(self) -> np.ndarray | None:
        """Entries pruned since the last call (for optimizer state resets)."""
        dropped, self._dropped = self._dropped, None
        return dropped

    @property
    def num_active(self) -> int:
        return self.size if self._mask is None else int(self._mask.sum())

    @property
    def sparsity(self) -> float:
        return 1.0 - self.num_active / self.size

    # --------------------------------------------------------------- forward
    def masked(self) -> Tensor:
        """The effective weight w * m, as seen by the forward pass."""
        if self._mask is None:
            return self
        m = self._mask.astype(self.dtype)

        def backward(g):
            self._dense_grad = g.copy() if self._dense_grad is None else self._dense_grad + g
            return (g * m,)

        return Tensor._make(self.data * m, (self,), backward, "mask")

    # ------------------------------------------------------------- gradients
    @property
    def dense_grad(self) -> np.ndarray | None:
        """dL/d(w * m); equals ``grad`` on active entries and for dense parameters."""
        return self.grad if self._mask is None else self._dense_grad

    def zero_grad(self) -> None:
        self.grad = None
        self._dense_grad = None

    def __repr__(self) -> str:
        name = f"{self.name!r}, " if self.name else ""
        return f"Parameter({name}shape={self.shape}, sparsity={self.sparsity:.3f})"
