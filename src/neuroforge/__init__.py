"""neuroforge - neural networks and CNNs from scratch with NumPy."""

from .activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from .augment import Augment
from .data import ImageDataset, Stats, batch_iterator
from .layers import (
    BatchNorm2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAvgPool2D,
    MaxPool2D,
)
from .losses import CrossEntropyLoss, MSELoss
from .model import Sequential, accuracy
from .optimizers import SGD, Adam, CosineAnnealingLR, StepLR

__all__ = [
    "SGD",
    "Adam",
    "Augment",
    "BatchNorm2D",
    "Conv2D",
    "CosineAnnealingLR",
    "CrossEntropyLoss",
    "Dense",
    "Dropout",
    "Flatten",
    "GlobalAvgPool2D",
    "ImageDataset",
    "LeakyReLU",
    "MSELoss",
    "MaxPool2D",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Stats",
    "StepLR",
    "Tanh",
    "accuracy",
    "batch_iterator",
]
