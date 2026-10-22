"""Composable layers built only on ``aqua.engine``.

Initialisation
--------------
For z = h W with W_ij ~ N(0, s^2) independent of h, E[z_j^2] = fan_in * s^2 * E[h_i^2].
To keep every pre-activation at E[z^2] = 1 (so neither the forward signal nor
the backward gradient grows or shrinks geometrically with depth), choose
s^2 = gain / fan_in, where the gain undoes whatever produced the layer's *input* h:

* Standardised data (the first layer): E[h^2] = 1, so gain 1.
* ReLU output: a symmetric z loses half its second moment, E[relu(z)^2] = E[z^2] / 2,
  so gain 2 (He et al., 2015). This includes the output layer, so logits also
  start at unit variance.
* GELU output: E[gelu(z)^2] = 0.4252 E[z^2] for z ~ N(0, 1) (numerical
  integration), so gain 1 / 0.4252 = 2.35. Using ReLU's 2 would lose 15% per layer.
* tanh output: tanh is the identity to first order around 0, so gain 1 (LeCun).
  It is mildly contractive, so the scale decays slowly with depth. That does not
  matter at 2-3 layers.

With unit-variance logits (variance s^2), a second-order expansion of
log-sum-exp gives an initial loss of about ln(C) + s^2 (C - 1) / (2C): close to
the uninformed ln(C), and never confidently wrong.

Biases start at 0. Inputs are standardised by the data loader, which is the
E[x^2] = 1 assumption the derivation starts from.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..engine import Parameter, Tensor

# Gain for a layer whose input is the output of this activation (see module docstring).
GAINS = {"input": 1.0, "relu": 2.0, "gelu": 2.35, "tanh": 1.0}


class Module:
    def parameters(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for value in vars(self).values():
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if isinstance(item, Parameter):
                    params = [item]
                elif isinstance(item, Module):
                    params = item.parameters()
                else:
                    continue
                for p in params:
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """y = x W + b, with W of shape (in, out). The weight enters through its mask."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator,
                 gain: float = 2.0, dtype=np.float32, name: str = "linear"):
        std = np.sqrt(gain / in_features)
        self.weight = Parameter(rng.normal(0.0, std, (in_features, out_features)).astype(dtype), name=f"{name}.weight")
        self.bias = Parameter(np.zeros(out_features, dtype=dtype), name=f"{name}.bias")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.masked() + self.bias


class Activation(Module):
    def __init__(self, kind: str):
        if kind not in ("relu", "tanh", "gelu"):
            raise ValueError(f"unknown activation {kind!r}")
        self.kind = kind

    def forward(self, x: Tensor) -> Tensor:
        return getattr(x, self.kind)()


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Sequential):
    def __init__(self, sizes: list[int], activation: str = "relu", seed: int = 0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        layers: list[Module] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            gain = GAINS["input"] if i == 0 else GAINS[activation]
            layers.append(Linear(fan_in, fan_out, rng, gain=gain, dtype=dtype, name=f"fc{i + 1}"))
            if not last:
                layers.append(Activation(activation))
        super().__init__(*layers)
        self.sizes = list(sizes)

    @property
    def linears(self) -> list[Linear]:
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    def prunable(self) -> list[Parameter]:
        """Weight matrices: the connections. Biases are not pruned (they are <0.2% of
        parameters and removing one shifts a whole unit, not a connection)."""
        return [layer.weight for layer in self.linears]
