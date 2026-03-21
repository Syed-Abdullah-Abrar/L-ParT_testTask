# Event Classification with Masked Transformer Autoencoders 

This repository is an update on Lorentz-ParT model. It performs test tasks for GSoC 2026 models updates.
## Updates include
- Using SDPA element wise gated attention (implemented).
- Implement Joint Embedding Predictive analysis (JEPA) for prdeiction in abstarct latent embedding. (not implemented)
- Using Open AI's Titron for writing multi GPU kernels. (not implemented)
- Abalation studies of the model. (not implemented)
- Extending models capabilities as to include particle mass regression. (not implemented) 


## Overview

High-energy physics jets are complex objects composed of many particles. This repo provides implementations of Particle Transformer-based models and training utilities to tackle:
- Jet classification (multi-class) using attention over per-particle inputs
- Masked particle reconstruction (predicting a masked particle’s kinematics) with conservation-aware losses


Core components:
- Models: `LorentzParT`, a `ParticleTransformer` with `EquiLinear` layers from `LGATr`.
- Engine: Training/eval loops, logging, checkpointing (see `src/engine/trainer.py`)
- Configs: YAML-driven experiment config (see `configs/`)
- Datasets: Utilities for loading the ROOT files from JetClass dataset (see `src/utils/data`)

## Get Started

### Prerequisites

- Python 3.13
- Git

### Installation

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv

    # On Windows
    .venv\Scripts\activate

    # On macOS/Linux
    source .venv/bin/activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyTorch (GPU or CPU):**

    See: https://pytorch.org/get-started/locally/

## Data

The JetClass dataset is publicly available: https://zenodo.org/records/6619768
This code has total of 100,000 tokens from JetClass dataset and placed under `./data/`
Since the .tar files at Zenodo had a minimum of 5M events, this repo developes a custom streaming script to dynamically pull and slice exactly 100,000 events (80k/10k/10k train/val/test splits) -npy files rather than ROOT- from Zenodo, avoiding the need to download the entire massive JetClass dataset locally. 

Structure

```
data/
├── train_80K/
│    ├── X_jets.npy
│    ├── X_particles.npy
│    └── y.npy
├── val_10k/
│   ├── X_jets.npy
│   ├── X_particles.npy
│   └── y.npy
└── test_20M/
    ├── X_jets.npy
    ├── X_particles.npy
    └── y.npy
```

## File changed in the test task 
Compared to the original https://github.com/ML4SCI/CMS/tree/main/MAEs/Hybrid_Transformer_Thanh_Nguyen, the following files were edited for the test tasks.
1. `src/models/particle_transformer.py` <br>
- To inject the NeurIPS 2025 SDPA Elementwise Gate, self.gate_proj (a linear layer) was added and the forward pass was modified to     multiply the scaled dot-product attention output by the sigmoid of the query projection (x = attn_out * gate).
2. `configs/train_100K_LorentzParT.yaml` (New file) <br>
- Copied from other `.yaml` files and set the pair_embed_dims mismatch, chnaged the batch_size to 30, and set weights: null to ensure the model initialized with fresh random weights
3. `scripts/evaluate_LorentzParT.py` <br>
- The original codebase unconditionally called torch.distributed.broadcast_object_list, which crashes on single-GPU/CPU setups. I wrapped it in an if torch.distributed.is_initialized(): safe check, allowing the evaluation script to run locally and generate the PNGs (The training was done single CPU system)
4. `src/utils/get_100k_dataset.py `(New Script) <br>
- Instead of downloading 100M+ events, this script dynamically streams the Zenodo tarball and extracts exactly an 80k/10k/10k split to strictly adhere to the GSoC Test Task constraints.


## Configuration

Experiments are defined in YAML. See `configs/train_100K_LorentzParT.yaml`

Model and training configs are parsed and fed into the trainers.

## Train and Evaluate

Two lightweight CLI scripts are provided under `scripts/` and use YAML experiment definitions:

- Training: `scripts/train_LorentzParT.py` — trains a `LorentzParT` model defined by the YAML config and saves logs/weights
- Evaluation: `scripts/evaluate_LorentzParT.py` — loads a trained model and runs evaluation/visualization on a test split


Train usage (example):

```bash
# single-GPU or CPU
python3 -m scripts.train_LorentzParT \
    --config-path ./configs/train_100K_LorentzParT.yaml \
    --train-data-dir ./data/train_80k \
    --val-data-dir ./data/val_10k
```

Key flags for `scripts/train_LorentzParT.py`:
- `--config-path`: Path to the YAML experiment 
- `--train-data-dir`: Directory with training npy (for test task only. Use ROOT otherwise) files
- `--val-data-dir`: Directory with validation npy (for test task only. Use ROOT otherwise) files


Evaluate usage (example):

```bash
# single-GPU or CPU
python3 -m scripts.evaluate_LorentzParT \
    --config-path ./configs/train_100K_LorentzParT.yaml \
    --best-model-path ./logs/LorentzParT/best/pid11245_20260319-060815.pt \
    --test-data-dir ./data/test_10k
```

Key flags for `scripts/evaluate.py`:
- `--config-path`: Path to the YAML experiment 
- `--best-model-path`: Path to saved best model weights used for evaluation (name for the .pt file can be specified in training script flag as checkpoint) 
- `--test-data-dir`: Directory with test npy (for test task only. Use ROOT otherwise) files


## Model and Losses

- `src/models/lorentz_part.py`: `ParticleTransformer` backbone and heads combined with `EquiLinear` layers from `LGATr`.
- `src/loss/`: Losses including `ConservationLoss` for masked reconstruction.



## Project Structure

```
Hybrid_Transformer_Thanh_Nguyen/
├── src/
│   ├── configs/           # Dataclasses and loaders for YAML configs
│   ├── engine/            # Training/evaluation engine
│   ├── loss/              # Loss functions and registry
│   ├── models/            # Particle Transformer and variants
│   ├── optim/             # Optimizers/schedulers registries
│   └── utils/             # Callbacks, metrics, data, viz helpers
├── scripts/               # CLI tools: train/evaluate
├── tests/                 # Unit tests
├── logs/                  # Runs: best weights and CSV logs
├── data/                  # ROOT files (train/val/test splits)
└── assets/                # Figures and demo assets
```
