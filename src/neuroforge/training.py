"""Training and evaluation loops."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from .data import iter_batches
from .model import Sequential, accuracy


def predict_logits(model: Sequential, x, batch_size: int = 128, tta: bool = False) -> np.ndarray:
    """Return per-sample logits, optionally with horizontal-flip TTA."""
    outputs = []
    for xb, _ in iter_batches(x, np.zeros(len(x), dtype=np.int64), batch_size, shuffle=False):
        logits = model.predict(xb)
        if tta:
            logits = 0.5 * (logits + model.predict(xb[:, :, :, ::-1]))
        outputs.append(logits)
    return np.concatenate(outputs, axis=0)


def evaluate(model: Sequential, loss_fn, x, y, batch_size: int = 128, tta: bool = False) -> tuple[float, float]:
    """Return ``(loss, accuracy)`` over ``x``/``y``.

    When ``tta`` is set, logits are averaged with those of the horizontally
    flipped image (test-time augmentation).
    """
    logits = predict_logits(model, x, batch_size, tta=tta)
    loss = loss_fn(logits, y)
    return loss, accuracy(logits, y)


def fit(
    model: Sequential,
    optimizer,
    loss_fn,
    train_data,
    val_data,
    epochs: int = 10,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    augment: Callable | None = None,
    scheduler=None,
    monitor: str = "loss",
    patience: int | None = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
) -> dict:
    """Run the training loop.

    ``train_data`` / ``val_data`` are ``(x, y)`` tuples of NumPy arrays.
    ``augment`` is an optional callable ``(x_batch, rng) -> x_batch`` applied
    to every training mini-batch. ``scheduler`` is stepped once per epoch.
    ``monitor`` selects the early-stopping metric (``"loss"`` or ``"acc"``).

    Returns a history dict with per-epoch metrics plus ``best_epoch``,
    ``best_val_loss``, ``best_val_acc`` and ``stopped_epoch``.
    """
    rng = rng or np.random.default_rng(0)
    x_train, y_train = train_data
    x_val, y_val = val_data
    history: dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_metric = -float("inf") if monitor == "acc" else float("inf")
    best_epoch = 0
    best_params = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_correct, seen = 0.0, 0, 0
        batches = iter_batches(x_train, y_train, batch_size, shuffle=True, rng=rng)
        if verbose:
            batches = tqdm(batches, total=int(np.ceil(len(x_train) / batch_size)), desc=f"epoch {epoch}/{epochs}", leave=False)
        for xb, yb in batches:
            if augment is not None:
                xb = augment(xb, rng)
            logits = model.forward(xb, training=True)
            loss, grad = loss_fn.forward(logits, yb)
            model.backward(grad)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss * len(yb)
            epoch_correct += accuracy(logits, yb) * len(yb)
            seen += len(yb)

        train_loss, train_acc = epoch_loss / seen, epoch_correct / seen
        val_loss, val_acc = evaluate(model, loss_fn, x_val, y_val, batch_size)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.lr)

        if monitor == "acc":
            current = val_acc
            improved = val_acc > best_metric + min_delta
        else:
            current = val_loss
            improved = val_loss < best_metric - min_delta

        if improved:
            best_metric = current
            best_epoch = epoch
            if restore_best:
                best_params = [p.copy() for p in model.params()]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}  "
                f"lr {optimizer.lr:.2e}"
            )

        if scheduler is not None:
            scheduler.step()

        if patience is not None and epochs_no_improve >= patience:
            stopped_epoch = epoch
            if verbose:
                print(f"early stopping at epoch {epoch} (no val {monitor} improvement for {patience} epochs)")
            break

    if restore_best and best_params is not None:
        for param, best in zip(model.params(), best_params):
            param[...] = best

    history["monitor"] = monitor
    history["best_epoch"] = best_epoch
    history["best_metric"] = best_metric
    history["best_val_loss"] = min(history["val_loss"])
    history["best_val_acc"] = max(history["val_acc"])
    history["stopped_epoch"] = stopped_epoch
    return history
