"""Loss functions implemented from scratch with NumPy."""

from __future__ import annotations

import numpy as np


class Loss:
    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        loss, _ = self.forward(logits, targets)
        return loss

    def forward(self, logits, targets):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Softmax cross-entropy.

    ``targets`` may be integer class indices of shape (N,) or one-hot
    vectors of shape (N, C). The returned gradient is with respect to the
    raw logits, i.e. ``(softmax(logits) - onehot) / N``.
    """

    def forward(self, logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        n = logits.shape[0]
        if targets.ndim == 1:
            onehot = np.zeros_like(probs)
            onehot[np.arange(n), targets] = 1.0
        else:
            onehot = targets
        eps = 1e-12
        loss = -np.sum(onehot * np.log(probs + eps)) / n
        self.grad = (probs - onehot) / n
        return float(loss), self.grad


class MSELoss(Loss):
    def forward(self, preds, targets):
        n = preds.shape[0]
        diff = preds - targets
        loss = np.sum(diff**2) / n
        self.grad = 2.0 * diff / n
        return float(loss), self.grad
