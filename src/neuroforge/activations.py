"""Elementwise activation functions implemented from scratch with NumPy.

Every activation caches whatever it needs in ``forward`` to compute the
gradient in ``backward``. All operations are vectorised and support
arbitrary leading batch dimensions.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class Activation:
    params: ClassVar[list[np.ndarray]] = []
    grads: ClassVar[list[np.ndarray]] = []

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x, training=True):
        self.x = x
        return np.where(x > 0, x, self.negative_slope * x)

    def backward(self, grad):
        return grad * np.where(self.x > 0, 1.0, self.negative_slope)


class Sigmoid(Activation):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh(Activation):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out**2)


class Softmax(Activation):
    """Numerically stable softmax over the last axis."""

    def forward(self, x, training=True):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.out

    def backward(self, grad):
        # Full Jacobian-vector product for a single distribution:
        # dL/dx = s * (dL/ds - sum(dL/ds * s))
        return self.out * (grad - np.sum(grad * self.out, axis=-1, keepdims=True))
