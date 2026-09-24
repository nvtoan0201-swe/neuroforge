"""Train a small CNN on the durian leaf dataset, from scratch.

Every run writes a full report (config, model, per-epoch history, test
results) to a single Markdown file so experiments are easy to review later.

Examples:
    uv run train_cnn.py --image-size 64 --augment --pool gap --width 32 \
        --lr-schedule cosine --epochs 40 --patience 8 --tta
    uv run train_cnn.py --image-size 32 --augment --dropout 0.3 --batchnorm \
        --seeds 0,1,2 --tta --run-name ensemble_demo
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neuroforge import (
    SGD,
    Adam,
    Augment,
    BatchNorm2D,
    Conv2D,
    CosineAnnealingLR,
    CrossEntropyLoss,
    Dense,
    Dropout,
    Flatten,
    GlobalAvgPool2D,
    ImageDataset,
    MaxPool2D,
    ReLU,
    Sequential,
    StepLR,
    accuracy,
)
from neuroforge.training import fit, predict_logits

DATA_ROOT = Path("data/Durian_Leaf_Diseases")
RUNS_DIR = Path("runs")
SUMMARY_NAME = "summary.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN from scratch.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--lr-schedule", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Cosine eta_min")
    parser.add_argument("--step-size", type=int, default=10, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR decay factor")

    parser.add_argument("--width", type=int, default=16, help="Base conv channels")
    parser.add_argument("--blocks", type=int, default=2, help="Number of conv blocks")
    parser.add_argument("--pool", choices=["flatten", "gap"], default="gap", help="Classifier head pooling")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--batchnorm", action="store_true", help="Add BatchNorm2D after each conv")

    parser.add_argument("--augment", action="store_true", help="Enable augmentation (hflip + crop)")
    parser.add_argument("--crop-pad", type=int, default=4, help="Random-crop padding when augmenting")
    parser.add_argument("--rotate90", action="store_true", help="Add random 90-degree rotations")
    parser.add_argument("--brightness", type=float, default=0.0, help="Random brightness delta")

    parser.add_argument("--patience", type=int, default=0, help="Early-stop patience (0 = off)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Min monitored improvement")
    parser.add_argument("--early-stop-metric", choices=["loss", "acc"], default="loss")
    parser.add_argument("--no-restore-best", action="store_false", dest="restore_best", help="Keep final weights")

    parser.add_argument("--tta", action="store_true", help="Horizontal-flip test-time augmentation")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for ensembling, e.g. 0,1,2")

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Explicit report path")
    return parser.parse_args()


def build_optimizer(args, model: Sequential):
    if args.optimizer == "adam":
        return Adam(
            model.params(),
            model.grads(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    return SGD(
        model.params(),
        model.grads(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.lr_schedule == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, t_max=args.epochs, eta_min=args.min_lr)
    return None


def build_model(
    num_classes: int,
    image_size: int,
    channels: int = 3,
    width: int = 16,
    blocks: int = 2,
    pool: str = "gap",
    dropout: float = 0.0,
    batchnorm: bool = False,
) -> Sequential:
    layers = []
    in_channels = channels
    for i in range(blocks):
        out_channels = width * (2**i)
        layers.append(Conv2D(in_channels, out_channels, 3, padding=1))
        if batchnorm:
            layers.append(BatchNorm2D(out_channels))
        layers.append(ReLU())
        layers.append(MaxPool2D(2))
        if dropout > 0.0:
            layers.append(Dropout(dropout))
        in_channels = out_channels
    layers.append(GlobalAvgPool2D() if pool == "gap" else Flatten())

    model = Sequential(layers)
    flat = model.predict(np.zeros((1, channels, image_size, image_size), dtype=np.float32)).shape[-1]
    model.add(Dense(flat, 128)).add(ReLU())
    if dropout > 0.0:
        model.add(Dropout(dropout))
    model.add(Dense(128, num_classes))
    return model


def build_augment(args) -> Augment | None:
    if not (args.augment or args.rotate90 or args.brightness):
        return None
    return Augment(
        horizontal_flip=args.augment,
        crop_pad=args.crop_pad if args.augment else 0,
        rotate90=args.rotate90,
        brightness=args.brightness,
    )


def default_run_name(args) -> str:
    parts = [args.optimizer, f"lr{args.lr:g}", f"img{args.image_size}", args.pool, f"w{args.width}", f"b{args.blocks}"]
    if args.batchnorm:
        parts.append("bn")
    if args.dropout:
        parts.append(f"do{args.dropout:g}")
    if args.augment:
        parts.append("aug")
    if args.rotate90:
        parts.append("rot")
    if args.brightness:
        parts.append(f"bri{args.brightness:g}")
    if args.lr_schedule != "none":
        parts.append(args.lr_schedule)
    if args.weight_decay:
        parts.append(f"wd{args.weight_decay:g}")
    if args.patience:
        parts.append(f"es{args.patience}")
    if args.tta:
        parts.append("tta")
    return "_".join(parts)


def history_table(history: dict) -> str:
    lines = [
        "| epoch | train loss | train acc | val loss | val acc | lr |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (tl, ta, vl, va, lr) in enumerate(
        zip(history["train_loss"], history["train_acc"], history["val_loss"], history["val_acc"], history["lr"]),
        start=1,
    ):
        lines.append(f"| {i} | {tl:.4f} | {ta:.4f} | {vl:.4f} | {va:.4f} | {lr:.2e} |")
    return "\n".join(lines)


def seed_section(seed: int, result: dict) -> list[str]:
    h = result["history"]
    return [
        f"### seed {seed}",
        "",
        f"- best val acc: {h['best_val_acc']:.4f}",
        f"- best val loss: {h['best_val_loss']:.4f} (epoch {h['best_epoch']})",
        f"- stopped epoch: {h['stopped_epoch']}",
        f"- test loss: {result['test_loss']:.4f}",
        f"- test acc: {result['test_acc']:.4f}",
        "",
        history_table(h),
        "",
    ]


def format_report(args, augment, classes, sizes, model, optimizer, scheduler, results, ensemble, run_name) -> str:
    opt_name = type(optimizer).__name__
    opt_info = f"{opt_name} (lr={args.lr:g}"
    if opt_name == "Adam":
        opt_info += f", betas=({args.beta1:g}, {args.beta2:g}), eps={args.eps:g}"
    else:
        opt_info += f", momentum={args.momentum:g}"
    opt_info += f", weight_decay={args.weight_decay:g})"
    es = f"{args.early_stop_metric}, patience={args.patience}, min_delta={args.min_delta}" if args.patience else "off"

    lines = [
        f"# Run: {run_name}",
        "",
        f"- timestamp: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- data root: `{args.data_root}`",
        f"- classes ({len(classes)}): {classes}",
        f"- sizes: train {sizes[0]} | val {sizes[1]} | test {sizes[2]}",
        f"- image size: {args.image_size} | batch size: {args.batch_size} | seeds: {[r['seed'] for r in results]}",
        (
            f"- architecture: width {args.width}, blocks {args.blocks}, pool {args.pool}, "
            f"batchnorm {args.batchnorm}, dropout {args.dropout:g}"
        ),
        f"- optimizer: {opt_info}",
        f"- lr schedule: {args.lr_schedule}"
        + (f" (min_lr={args.min_lr:g}, step_size={args.step_size}, gamma={args.gamma:g})" if args.lr_schedule != "none" else ""),
        f"- augment: {augment.describe() if augment else 'none'}",
        f"- early stop: {es} | restore best: {args.restore_best}",
        f"- tta: {args.tta}",
        "",
        "## Model",
        "",
        "```",
        model.summary(),
        "```",
        "",
    ]

    if ensemble is not None:
        lines += [
            "## Ensemble result",
            "",
            f"- test loss: {ensemble['loss']:.4f}",
            f"- test acc: {ensemble['acc']:.4f}",
            "",
        ]

    lines.append("## Per-seed results")
    lines.append("")
    for result in results:
        lines += seed_section(result["seed"], result)

    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def append_summary(path: Path, rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| run | seed | optimizer | lr | schedule | pool | width | blocks | dropout | augment | tta | "
        "epochs run | best val acc | test loss | test acc |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not path.exists():
        path.write_text(header)
    with path.open("a") as f:
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def run_once(seed, args, loss_fn, classes, train, val, test, augment):
    np.random.seed(seed)
    model = build_model(
        len(classes),
        args.image_size,
        width=args.width,
        blocks=args.blocks,
        pool=args.pool,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
    )
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    history = fit(
        model,
        optimizer,
        loss_fn,
        (train.images, train.labels),
        (val.images, val.labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=augment,
        scheduler=scheduler,
        monitor=args.early_stop_metric,
        patience=args.patience or None,
        min_delta=args.min_delta,
        restore_best=args.restore_best,
    )
    logits = predict_logits(model, test.images, args.batch_size, tta=args.tta)
    return {
        "seed": seed,
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "logits": logits,
        "test_loss": float(loss_fn(logits, test.labels)),
        "test_acc": float(accuracy(logits, test.labels)),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    size = (args.image_size, args.image_size)
    if args.image_size % (2**args.blocks) != 0:
        raise SystemExit(f"--image-size must be divisible by 2**blocks = {2**args.blocks}")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    run_name = args.run_name or default_run_name(args)
    augment = build_augment(args)

    raw = ImageDataset(args.data_root / "train", image_size=size)
    stats = raw.compute_stats()
    classes = raw.classes
    train = ImageDataset(args.data_root / "train", image_size=size, classes=classes, stats=stats)
    val = ImageDataset(args.data_root / "val", image_size=size, classes=classes, stats=stats)
    test = ImageDataset(args.data_root / "test", image_size=size, classes=classes, stats=stats)
    print(f"run: {run_name} | seeds: {seeds}")
    print(f"classes: {classes}")
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    loss_fn = CrossEntropyLoss()
    results = []
    for seed in seeds:
        result = run_once(seed, args, loss_fn, classes, train, val, test, augment)
        results.append(result)
        print(f"seed {seed}: test loss {result['test_loss']:.4f} acc {result['test_acc']:.4f}")

    ensemble = None
    if len(results) > 1:
        mean_logits = np.mean([r["logits"] for r in results], axis=0)
        ensemble = {
            "loss": float(loss_fn(mean_logits, test.labels)),
            "acc": float(accuracy(mean_logits, test.labels)),
        }
        print(f"ensemble: test loss {ensemble['loss']:.4f} acc {ensemble['acc']:.4f}")

    report = format_report(
        args, augment, classes, (len(train), len(val), len(test)),
        results[-1]["model"], results[-1]["optimizer"], None, results, ensemble, run_name,
    )
    report_path = args.output or (args.runs_dir / f"{run_name}.md")
    write_report(report_path, report)

    rows = []
    for result in results:
        h = result["history"]
        rows.append((
            run_name, str(result["seed"]), args.optimizer, f"{args.lr:g}", args.lr_schedule,
            args.pool, str(args.width), str(args.blocks), f"{args.dropout:g}",
            "yes" if args.augment else "no", "yes" if args.tta else "no",
            str(len(h["train_loss"])), f"{h['best_val_acc']:.4f}", f"{result['test_loss']:.4f}", f"{result['test_acc']:.4f}",
        ))
    if ensemble is not None:
        best_val = max(r["history"]["best_val_acc"] for r in results)
        rows.append((
            run_name, "ensemble", args.optimizer, f"{args.lr:g}", args.lr_schedule,
            args.pool, str(args.width), str(args.blocks), f"{args.dropout:g}",
            "yes" if args.augment else "no", "yes" if args.tta else "no",
            str(max(len(r["history"]["train_loss"]) for r in results)), f"{best_val:.4f}",
            f"{ensemble['loss']:.4f}", f"{ensemble['acc']:.4f}",
        ))
    append_summary(args.runs_dir / SUMMARY_NAME, rows)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
