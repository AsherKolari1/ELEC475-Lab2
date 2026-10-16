# ELEC 475 Lab 2
## From LeNet-5 to AlexNet: An Architectural & Empirical Evolution

Completed implementation and experimental results for ELEC 475 Lab 2.

This repository evaluates Base LeNet-5, Modernized LeNet, custom AlexNet-64 variants, data augmentation, PyTorch Zoo AlexNet, CIFAR-10, Tiny ImageNet, and an extended multi-seed batch-size ablation study.

## Environment Setup

### Linux / WSL / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the Experiments

The exact CLI commands used for the experiments are provided in:

```text
commands.txt
```

The commands are organized into:

1. CIFAR-10 master model runs
2. Data augmentation comparison
3. Tiny ImageNet master model runs
4. Extended multi-seed batch-size ablation
5. Required EDA, transform, profiling, and visualization utilities

Commands in `commands.txt` are intended to be run individually from the repository root.

## Main Models

The repository contains implementations of:

- Base LeNet-5
- Modernized LeNet
- AlexNet-64 with no normalization
- AlexNet-64 with Local Response Normalization (LRN)
- AlexNet-64 with Batch Normalization
- AlexNet-64 with Batch Normalization and data augmentation
- PyTorch Model Zoo AlexNet adapted for the lab datasets

## Datasets

Experiments were performed on:

- CIFAR-10
- Tiny ImageNet-200

Dataset loading and preprocessing are handled through `utils/dataset.py`.

Dataset files are not stored in the repository.

## Extended Ablation Study

An additional multi-seed experiment evaluated the effect of batch size on AlexNet-64 with Batch Normalization using batch sizes:

```text
64, 128, 256
```

with seeds:

```text
17, 73, 149
```

The corresponding experimental logs are stored under:

```text
results/extended_ablation/
```

## Model Checkpoints

Model checkpoints are stored in:

```text
checkpoints/
```

Large checkpoint files are managed using Git LFS.
