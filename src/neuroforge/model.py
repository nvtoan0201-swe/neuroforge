"""A minimal sequential model container built on the from-scratch layers."""

from __future__ import annotations

import numpy as np

from .layers import Layer


class Sequential:
    def __init__(self, layers=None):
        self.layers: list[Layer] = list(layers) if layers else []

    def add(self, layer: Layer) -> Sequential:
        self.layers.append(layer)
        return self

    def forward(self, x, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x) -> np.ndarray:
        return self.forward(x, training=False)

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads]

    def summary(self) -> str:
        lines = [f"{'Layer':<22}{'Param shape':<24}{'#Params':>10}"]
        for layer in self.layers:
            name = type(layer).__name__
            if layer.params:
                shapes = ", ".join(str(p.shape) for p in layer.params)
                count = sum(p.size for p in layer.params)
            else:
                shapes, count = "-", 0
            lines.append(f"{name:<22}{shapes:<24}{count:>10}")
        lines.append(f"{'Total':<46}{sum(p.size for p in self.params()):>10}")
        return "\n".join(lines)


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=-1)
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=-1)
    return float(np.mean(preds == targets))
