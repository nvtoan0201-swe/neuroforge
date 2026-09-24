# neuroforge

Convolutional neural networks implemented **from scratch with NumPy** — no
PyTorch, TensorFlow, or autograd engine. Every layer, loss, optimiser and
augmentation is written by hand with explicit forward/backward passes, then
trained on the **Durian Leaf Diseases** image-classification dataset.

The goal of this project is to make the mathematics behind CNNs concrete:
you can read exactly what a convolution, a pooling layer or Adam does, line
by line.

## What's implemented

| Area | Components |
|---|---|
| Layers | `Conv2D` (im2col), `MaxPool2D`, `GlobalAvgPool2D`, `Dense`, `Flatten`, `BatchNorm2D`, `Dropout` |
| Activations | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax` |
| Losses | `CrossEntropyLoss`, `MSELoss` |
| Optimisers | `SGD` (momentum + weight decay), `Adam` |
| LR schedules | `StepLR`, `CosineAnnealingLR` |
| Augmentation | horizontal flip, random crop, 90° rotation, brightness |
| Training | `fit()` / `evaluate()`, early stopping, best-weight restore, hflip TTA, multi-seed ensembling |
| Model | `Sequential` container with parameter summary |

## Dataset

**A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural
Diagnosis** — 6 classes of durian leaf conditions:

```
Leaf_Algal  Leaf_Blight  Leaf_Colletotrichum
Leaf_Healthy  Leaf_Phomopsis  Leaf_Rhizoctonia
```

The dataset is committed in this repository under
`data/Durian_Leaf_Diseases/`, split as:

| split | images |
|---|---:|
| train | 1814 |
| val | 387 |
| test | 394 |

Layout expected by the loader:

```
data/Durian_Leaf_Diseases/<split>/<class>/*.jpg
```

Images are decoded with Pillow, resized, scaled to `[0, 1]` and standardised
using the per-channel mean/std computed on the **train** split.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) — the only thing you need to install.
  All Python dependencies are pinned in `pyproject.toml` / `uv.lock`.
- Python **3.14+**. You do **not** need to install Python yourself: `uv`
  reads `.python-version` and downloads a matching interpreter automatically.

## Setup

```bash
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code
git clone https://github.com/nvtoan0201-swe/neuroforge.git
cd neuroforge

# 3. Create the virtualenv and install dependencies
uv sync
```

That's it — the dataset is committed in the repo, so you can train right
away:

```bash
uv run train_cnn.py --image-size 32 --blocks 2 --epochs 1   # quick smoke test
```

## Usage

Train the CNN (defaults: 30 epochs, image size 64, Adam):

```bash
uv run train_cnn.py --image-size 64 --batch-size 32
```

Best configuration found so far (96×96, gap head, batch norm, cosine LR,
early stopping, TTA):

```bash
uv run train_cnn.py \
  --image-size 96 --batch-size 32 --epochs 30 \
  --optimizer adam --lr 1e-3 --lr-schedule cosine \
  --width 32 --blocks 3 --pool gap --batchnorm \
  --augment --crop-pad 4 --patience 8 --tta
```

Multi-seed ensemble:

```bash
uv run train_cnn.py --image-size 64 --augment --batchnorm \
  --seeds 0,1,2 --tta --run-name ensemble_demo
```

### `train_cnn.py` options

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `data/Durian_Leaf_Diseases` | Dataset root |
| `--epochs` | `30` | Max epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--image-size` | `64` | Square input size; must be divisible by `2**blocks` |
| `--seed` | `0` | Random seed |
| `--optimizer` | `adam` | `adam` or `sgd` |
| `--lr` | `1e-3` | Learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight-decay` | `0.0` | L2 weight decay |
| `--beta1` / `--beta2` / `--eps` | `0.9` / `0.999` / `1e-8` | Adam hyper-parameters |
| `--lr-schedule` | `none` | `none`, `step` or `cosine` |
| `--min-lr` | `0.0` | Cosine `eta_min` |
| `--step-size` / `--gamma` | `10` / `0.1` | StepLR parameters |
| `--width` | `16` | Base number of conv channels (doubles each block) |
| `--blocks` | `2` | Number of conv blocks |
| `--pool` | `gap` | Classifier head: `gap` or `flatten` |
| `--dropout` | `0.0` | Dropout probability |
| `--batchnorm` | off | Add `BatchNorm2D` after each conv |
| `--augment` | off | Horizontal flip + random crop |
| `--crop-pad` | `4` | Crop padding when augmenting |
| `--rotate90` | off | Add random 90° rotations |
| `--brightness` | `0.0` | Random brightness delta |
| `--patience` | `0` | Early-stopping patience (0 = off) |
| `--min-delta` | `0.0` | Minimum monitored improvement |
| `--early-stop-metric` | `loss` | `loss` or `acc` |
| `--no-restore-best` | off | Keep final weights instead of best |
| `--tta` | off | Horizontal-flip test-time augmentation |
| `--seeds` | — | Comma-separated seeds for ensembling, e.g. `0,1,2` |
| `--run-name` | auto | Report file name |
| `--runs-dir` | `runs` | Directory for reports |
| `--output` | — | Explicit report path |

Every run writes a Markdown report (config, model summary, per-epoch history,
test results) to `runs/<run-name>.md` and appends a row to
`runs/summary.md`.

## Results

Test accuracy on the held-out test split (394 images). The test set is
evaluated **once** after training, never used to update weights.

| run | image | head | width | blocks | dropout | test acc |
|---|---:|---|---:|---:|---:|---:|
| `s96_gap_w32b3` | 96 | gap | 32 | 3 | 0.0 | **0.8528** |
| `s96_gap_w32b3_rot` | 96 | gap | 32 | 3 | 0.0 | 0.8426 |
| `s96_gap_w32b3_bri` | 96 | gap | 32 | 3 | 0.0 | 0.8325 |
| `s64_gap_w32b3` | 64 | gap | 32 | 3 | 0.0 | 0.8249 |
| `s32_gap_w32b3` | 32 | gap | 32 | 3 | 0.0 | 0.7513 |
| `s32_flat_w16b2` | 32 | flatten | 16 | 2 | 0.0 | 0.7030 |
| `s96_gap_w32b3_do0.3` | 96 | gap | 32 | 3 | 0.3 | 0.6827 |
| `s96_gap_w32b3_full` | 96 | gap | 32 | 3 | 0.3 | 0.6294 |
| `s32_gap_w16b2` | 32 | gap | 16 | 2 | 0.0 | 0.5812 |

Key findings:

- **Input resolution matters most**: 32 → 64 → 96 monotonically improves the
  `gap`/`w32`/`b3` model (0.7513 → 0.8249 → 0.8528).
- **Global average pooling beats flatten** with far fewer parameters
  (0.7513 vs 0.7030 at 32×32).
- **Dropout 0.3 hurts** this small model at 96×96 (0.8528 → 0.6827); extra
  rotation/brightness augmentation also gives no gain over plain
  hflip + crop.
- An earlier optimizer sweep found `Adam lr=1e-3` clearly ahead of `lr=1e-2`
  and SGD variants on the original 32×32 flatten model.

Full per-epoch logs live in [`runs/`](runs/); the table index is
[`runs/summary.md`](runs/summary.md).

## Repository layout

```
src/neuroforge/
  activations.py   ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
  layers.py        Conv2D (im2col), MaxPool2D, Dense, Flatten,
                   GlobalAvgPool2D, BatchNorm2D, Dropout
  losses.py        CrossEntropyLoss, MSELoss
  optimizers.py    SGD (momentum), Adam, StepLR, CosineAnnealingLR
  model.py         Sequential container + accuracy
  data.py          ImageDataset loader + batching
  augment.py       NumPy image augmentations
  training.py      fit() / evaluate() loops
train_cnn.py       training entry point + experiment reports
runs/              per-experiment Markdown reports + summary.md
data/              Durian Leaf Diseases dataset
```

## Development

```bash
uv run ruff check .    # lint
```

## Notes

- Everything runs on **CPU with NumPy**. There is no CUDA/GPU backend, so
  training time scales with image resolution.
- `--image-size` must be divisible by `2**blocks` (two 2×2 pools per block
  by default) or the script exits with an error.
