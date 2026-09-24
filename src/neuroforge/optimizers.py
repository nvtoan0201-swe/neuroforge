"""Gradient-based optimisers implemented from scratch with NumPy."""

from __future__ import annotations

import math

import numpy as np


class Optimizer:
    def __init__(self, params, grads, lr=1e-3):
        self.params = params
        self.grads = grads
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0.0


class SGD(Optimizer):
    def __init__(self, params, grads, lr=1e-2, momentum=0.9, weight_decay=0.0):
        super().__init__(params, grads, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p) for p in params]

    def step(self):
        for i, (param, grad) in enumerate(zip(self.params, self.grads)):
            g = grad
            if self.weight_decay:
                g = g + self.weight_decay * param
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * g
            param += self.velocity[i]


class Adam(Optimizer):
    def __init__(self, params, grads, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, grads, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for i, (param, grad) in enumerate(zip(self.params, self.grads)):
            g = grad + self.weight_decay * param if self.weight_decay else grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g**2
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LRScheduler:
    def __init__(self, optimizer: Optimizer, base_lr: float | None = None):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr if base_lr is None else base_lr
        self.last_epoch = 0

    def step(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, base_lr: float | None = None):
        super().__init__(optimizer, base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr = self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, t_max: int, eta_min: float = 0.0, base_lr: float | None = None):
        super().__init__(optimizer, base_lr)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.t_max)
        )
