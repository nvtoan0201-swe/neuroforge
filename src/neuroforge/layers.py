"""Neural network layers implemented from scratch with NumPy.

Every layer exposes three attributes used by the optimiser:

* ``params`` - list of trainable arrays
* ``grads``  - list of gradients aligned with ``params``

``forward`` accepts a single tensor and ``backward`` receives the gradient
of the loss with respect to that layer's output.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


class Layer:
    params: ClassVar[list[np.ndarray]] = []
    grads: ClassVar[list[np.ndarray]] = []

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x, training: bool = True):
        return self.forward(x, training)


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.weight = _he_normal(in_features, out_features)
        self.bias = np.zeros(out_features)
        self.params = [self.weight, self.bias]
        self.grads = [np.zeros_like(self.weight), np.zeros_like(self.bias)]

    def forward(self, x, training=True):
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, grad):
        self.grads[0][...] = self.x.reshape(-1, self.x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        self.grads[1][...] = grad.reshape(-1, grad.shape[-1]).sum(axis=0)
        return grad @ self.weight.T


class Flatten(Layer):
    def forward(self, x, training=True):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


def _im2col(x, kh, kw, stride, pad):
    """Turn (N, C, H, W) into (N*out_h*out_w, C*kh*kw) patch matrix."""
    n, c, h, w = x.shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, c * kh * kw)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride, pad):
    """Scatter-add a patch matrix back into an image-shaped gradient."""
    n, c, h, w = x_shape
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1
    cols = cols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    x_padded = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded


class Conv2D(Layer):
    """2D convolution supporting stride and zero padding (im2col based)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kh, self.kw = _pair(kernel_size)
        self.stride = _pair(stride)[0]
        self.padding = _pair(padding)[0]
        fan_in = in_channels * self.kh * self.kw
        self.weight = np.random.randn(out_channels, in_channels, self.kh, self.kw) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(out_channels)
        self.params = [self.weight, self.bias]
        self.grads = [np.zeros_like(self.weight), np.zeros_like(self.bias)]

    def forward(self, x, training=True):
        self.x_shape = x.shape
        n = x.shape[0]
        out_channels = self.weight.shape[0]
        cols, out_h, out_w = _im2col(x, self.kh, self.kw, self.stride, self.padding)
        self.cols = cols
        self.out_hw = (out_h, out_w)
        out = cols @ self.weight.reshape(out_channels, -1).T + self.bias
        return out.reshape(n, out_h, out_w, out_channels).transpose(0, 3, 1, 2)

    def backward(self, grad):
        out_channels = self.weight.shape[0]
        grad_flat = grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        self.grads[0][...] = (grad_flat.T @ self.cols).reshape(self.weight.shape)
        self.grads[1][...] = grad_flat.sum(axis=0)
        dcols = grad_flat @ self.weight.reshape(out_channels, -1)
        return _col2im(dcols, self.x_shape, self.kh, self.kw, self.stride, self.padding)


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = _pair(pool_size)
        self.stride = _pair(stride)[0] if stride is not None else self.pool_size[0]

    def forward(self, x, training=True):
        n, c, h, w = x.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.x_shape = x.shape
        self.out_hw = (out_h, out_w)
        out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
        self.argmax = np.empty((n, c, out_h, out_w), dtype=np.int64)
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                window = x[:, :, hs:hs + ph, ws:ws + pw].reshape(n, c, ph * pw)
                idx = window.argmax(axis=2)
                self.argmax[:, :, i, j] = idx
                out[:, :, i, j] = np.take_along_axis(window, idx[:, :, None], axis=2)[:, :, 0]
        return out

    def backward(self, grad):
        n, c, h, w = self.x_shape
        pw = self.pool_size[1]
        out_h, out_w = self.out_hw
        rows = self.argmax // pw + np.arange(out_h)[None, None, :, None] * self.stride
        cols = self.argmax % pw + np.arange(out_w)[None, None, None, :] * self.stride
        base = np.arange(n)[:, None, None, None] * c + np.arange(c)[None, :, None, None]
        flat = ((base * h + rows) * w + cols).ravel()
        dx = np.bincount(flat, weights=grad.ravel(), minlength=n * c * h * w)
        return dx.reshape(self.x_shape).astype(grad.dtype, copy=False)


class BatchNorm2D(Layer):
    """Batch normalisation over the (N, H, W) axes for each channel.

    Keeps running estimates of mean/variance so evaluation can use the
    population statistics collected during training.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

    def forward(self, x, training=True):
        _, c, _, _ = x.shape
        axes = (0, 2, 3)
        if training:
            mean = x.mean(axis=axes)
            var = x.var(axis=axes)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var
        self.x_shape = x.shape
        self.axes = axes
        self.mean = mean.reshape(1, c, 1, 1)
        self.std_inv = 1.0 / np.sqrt(var.reshape(1, c, 1, 1) + self.eps)
        self.x_norm = (x - self.mean) * self.std_inv
        return self.gamma.reshape(1, c, 1, 1) * self.x_norm + self.beta.reshape(1, c, 1, 1)

    def backward(self, grad):
        n, _, h, w = self.x_shape
        m = n * h * w
        self.grads[0][...] = np.sum(grad * self.x_norm, axis=self.axes)
        self.grads[1][...] = np.sum(grad, axis=self.axes)
        grad_norm = grad * self.gamma.reshape(1, -1, 1, 1)
        return self.std_inv / m * (
            m * grad_norm
            - grad_norm.sum(axis=self.axes, keepdims=True)
            - self.x_norm * np.sum(grad_norm * self.x_norm, axis=self.axes, keepdims=True)
        )


class GlobalAvgPool2D(Layer):
    """Average each channel over its spatial dimensions: (N,C,H,W) -> (N,C)."""

    def forward(self, x, training=True):
        self.x_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad):
        _, _, h, w = self.x_shape
        return np.broadcast_to(grad[:, :, None, None], self.x_shape) / (h * w)


class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        self.p = p

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            return x
        self.mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask
