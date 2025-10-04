# Facial Emotion Recognition (FER)

Recognize facial emotions — **happy**, **neutral**, **sad**, **surprise** — from grayscale/colored face images using PyTorch.
This repository is organized for reproducible experiments with clear configs, modular code, and clean outputs.

> Dataset folders are already provided in `data/` as three splits: `train/`, `validation/`, `test/`, each containing subfolders:
> `happy/`, `neutral`, `sad`, `surprise/`.

---

## 1) Project layout

```
facial-emotion-recognition/
├─ configs/                 # YAML configs for experiments
│   ├─ baseline_cnn.yaml
│   └─ resnet18.yaml
├─ data/                    # dataset (not tracked by git)
│   ├─ train/{happy,neutral,sad,surprise}/
│   ├─ validation/{happy,neutral,sad,surprise}/
│   └─ test/{happy,neutral,sad,surprise}/
├─ notebooks/               # EDA & quick experiments
├─ reports/                 # figures, exported charts (gitignored)
├─ runs/                    # logs, checkpoints, metrics (gitignored)
├─ src/                     # source code (importable as a package)
│   ├─ data/
│   │   ├─ datamodule.py    # Dataset & DataLoaders
│   │   └─ transforms.py    # Albumentations / Torch transforms
│   ├─ models/
│   │   ├─ baseline_cnn.py  # Small baseline CNN
│   │   ├─ heads.py         # Classifier heads
│   │   └─ utils.py         # Helpers (seeding, metrics, etc.)
│   ├─ train.py             # Training entrypoint
│   ├─ evaluate.py          # Evaluation & reports (CLS report, CM)
│   └─ inference.py         # Single image / folder inference
├─ .gitattributes           # Git LFS rules for model weights
├─ .gitignore
├─ .pre-commit-config.yaml  # Formatting & lint hooks
├─ pyproject.toml           # Tooling configs (black, isort, ruff)
├─ requirements.txt
└─ README.md
```

---

## 2) Quick start

### Create & activate virtual env
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS / Linux:
source .venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### (Optional, recommended) Git LFS for model weights
```bash
git lfs install
# *.pt / *.pth / *.ckpt are already routed via LFS in .gitattributes
```

### (Optional) Pre-commit hooks for formatting/linting
```bash
pip install pre-commit
pre-commit install
```

---

## 3) Configuration

All training hyperparameters live in YAML files in `configs/`. Example keys you might see:

```yaml
# configs/baseline_cnn.yaml
seed: 42
data:
  train_dir: data/train
  val_dir: data/validation
  test_dir: data/test
  img_size: 224
  batch_size: 64
  num_workers: 4
model:
  name: baseline_cnn
  num_classes: 4
  dropout: 0.25
optim:
  name: adam
  lr: 3e-4
  weight_decay: 1e-4
train:
  epochs: 25
  mixed_precision: false
  early_stopping: true
  checkpoint_dir: runs/baseline_cnn/
```

Adjust paths, image size, augmentations, and optimizer settings to your needs.

---

## 4) Training

```bash
python -m src.train --config configs/baseline_cnn.yaml
# or
python -m src.train --config configs/resnet18.yaml
```

Artifacts (checkpoints, logs, metrics) are written under `runs/<exp_name>/`.

---

## 5) Evaluation

Run on the **test** split and save a classification report + confusion matrix:

```bash
python -m src.evaluate \
  --ckpt runs/baseline_cnn/best.pt \
  --config configs/baseline_cnn.yaml
```

Outputs (reports, figures) are stored under `reports/` and/or alongside the run directory.

---

## 6) Inference

Predict a single image or all images in a folder:

```bash
# single image
python -m src.inference \
  --ckpt runs/baseline_cnn/best.pt \
  --image path/to/face.jpg \
  --config configs/baseline_cnn.yaml

# folder
python -m src.inference \
  --ckpt runs/baseline_cnn/best.pt \
  --folder path/to/folder \
  --config configs/baseline_cnn.yaml
```

---

## 7) Notes & tips

- **Class balance**: if classes are imbalanced, consider class weights or focal loss.
- **Augmentations**: start simple (flip, rotate, brightness/contrast) and tune gradually.
- **Reproducibility**: set seeds & deterministic flags (already included in utils).
- **Model size**: prefer Git LFS for pushing checkpoints to GitHub.
- **Hardware**: scripts auto-detect GPU/CPU; control mixed precision via config.

---

## 8) License / Acknowledgements

- Dataset provided as part of the assignment; please respect its license/usage terms.
- Code in this repository is released for educational purposes.

---

**Happy training!** 🚀
