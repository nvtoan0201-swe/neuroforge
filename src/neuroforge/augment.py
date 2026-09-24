"""Batch-level image augmentations implemented from scratch with NumPy.

Every function takes a float32 tensor shaped ``(N, C, H, W)`` plus a
:class:`numpy.random.Generator` and returns an augmented copy. Randomness is
per-sample so a whole mini-batch can be transformed in one call.
"""

from __future__ import annotations

import numpy as np


def random_horizontal_flip(x: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    if p <= 0.0:
        return x
    flip = rng.random(x.shape[0]) < p
    if not flip.any():
        return x
    out = x.copy()
    out[flip] = out[flip][:, :, :, ::-1]
    return out


def random_crop(x: np.ndarray, rng: np.random.Generator, pad: int = 4) -> np.ndarray:
    if pad <= 0:
        return x
    n, _, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    out = np.empty_like(x)
    for i in range(n):
        top = int(rng.integers(0, 2 * pad + 1))
        left = int(rng.integers(0, 2 * pad + 1))
        out[i] = padded[i, :, top:top + h, left:left + w]
    return out


def random_rotate90(x: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    if p <= 0.0:
        return x
    out = x.copy()
    for i in range(x.shape[0]):
        if rng.random() < p:
            out[i] = np.rot90(out[i], int(rng.integers(1, 4)), axes=(1, 2))
    return out


def random_brightness(x: np.ndarray, rng: np.random.Generator, max_delta: float = 0.05) -> np.ndarray:
    if max_delta <= 0.0:
        return x
    delta = rng.uniform(-max_delta, max_delta, size=(x.shape[0], 1, 1, 1)).astype(x.dtype)
    return x + delta


class Augment:
    """Compose a set of augmentations, applied in a fixed order."""

    def __init__(
        self,
        horizontal_flip: bool = True,
        crop_pad: int = 0,
        rotate90: bool = False,
        brightness: float = 0.0,
    ):
        self.horizontal_flip = horizontal_flip
        self.crop_pad = crop_pad
        self.rotate90 = rotate90
        self.brightness = brightness

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.horizontal_flip:
            x = random_horizontal_flip(x, rng)
        if self.crop_pad:
            x = random_crop(x, rng, self.crop_pad)
        if self.rotate90:
            x = random_rotate90(x, rng)
        if self.brightness:
            x = random_brightness(x, rng, self.brightness)
        return x

    def describe(self) -> str:
        parts = []
        if self.horizontal_flip:
            parts.append("hflip")
        if self.crop_pad:
            parts.append(f"crop(pad={self.crop_pad})")
        if self.rotate90:
            parts.append("rotate90")
        if self.brightness:
            parts.append(f"brightness({self.brightness})")
        return " + ".join(parts) if parts else "none"
