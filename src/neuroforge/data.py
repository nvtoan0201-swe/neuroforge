"""Dataset loading and batching utilities for image folders.

Expects the layout ``root/<split>/<class>/*.jpg`` or ``root/<class>/*.jpg``.
Images are decoded with Pillow, resized, converted to float32 in [0, 1] and
optionally standardised per channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Stats:
    mean: np.ndarray  # (C, 1, 1)
    std: np.ndarray  # (C, 1, 1)


class ImageDataset:
    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (32, 32),
        classes: list[str] | None = None,
        stats: Stats | None = None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {self.root}")

        if classes is None:
            classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.classes = classes
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.image_size = tuple(image_size)
        self.stats = stats

        self.images, self.labels, self.paths = self._load()

    def _load(self):
        images, labels, paths = [], [], []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(path) as img:
                    img = img.convert("RGB").resize(self.image_size, Image.BILINEAR)
                    array = np.asarray(img, dtype=np.float32) / 255.0
                images.append(array.transpose(2, 0, 1))  # HWC -> CHW
                labels.append(self.class_to_idx[class_name])
                paths.append(path)
        if not images:
            raise RuntimeError(f"no images found under {self.root}")
        x = np.stack(images).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if self.stats is not None:
            x = (x - self.stats.mean) / self.stats.std
        return x, y, paths

    def compute_stats(self) -> Stats:
        mean = self.images.mean(axis=(0, 2, 3), keepdims=True)
        std = self.images.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        return Stats(mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def iter_batches(x, y, batch_size, shuffle=True, rng=None):
    """Yield ``(x, y)`` mini-batches from arrays."""
    n = len(x)
    indices = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng()).shuffle(indices)
    for start in range(0, n, batch_size):
        batch = indices[start:start + batch_size]
        yield x[batch], y[batch]


def batch_iterator(dataset, batch_size, shuffle=True, rng=None):
    """Yield ``(x, y)`` mini-batches from an :class:`ImageDataset`."""
    return iter_batches(dataset.images, dataset.labels, batch_size, shuffle, rng)
