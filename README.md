# Gomoku Neural Network Training Pipeline

This repository implements the neural network training pipeline for a state-of-the-art Gomoku (Five in a Row) agent. The architecture and training methodologies are designed based on the **Rapfi** paper and incorporate advanced positional features, multi-task auxiliary heads, tactical edge-resampling, and D4 symmetry augmentations.

## 📖 Paper & Architecture Basis

This project is built around the techniques described in the **Rapfi** Gomoku paper:
- **Feature representation**: Input planes explicitly track board positions relative to the active player, supplemented by CoordConv coordinates and boundary-distance grids to resolve edge insensitivity.
- **Learnable Position Embeddings**: A grid-decomposed positional embedding layer is added directly to the network's latent layers.
- **Multi-Task Auxiliary Heads**: Alongside the primary policy and value heads, the model trains on auxiliary prediction tasks (Threat Type, Edge Threat Focus, Distance-to-Win, and Legal Moves). These heads guide the network to learn geometric and strategic structures, and are discarded during inference to maintain high speed.

---

## 📊 Datasets

The pipeline combines multiple high-quality datasets to prevent overfitting and policy collapse:

1. **WinePy Self-Play Dataset (`dataset/gomoku_dataset_split/`)**
   - **Source**: Equivalent to the Hugging Face [Karesis/Gomoku](https://huggingface.co/datasets/Karesis/Gomoku) dataset.
   - **Scale**: 875 self-played games, resulting in 26,378 unique training states.
   - **Generation**: Created using the WinePy alpha-beta search algorithm (searching 4–10 plies).
2. **Gomocup 2017 Tournament Dataset (`dataset/gomocup2017/`)**
   - **Scale**: 25,848 game files in `.psq` format.
   - **Composition**: Fastgame (15,984 files), Freestyle (8,280 files), Renju (504 files), Standard (1,080 files).
   - **Role**: Provides diverse strategies and rules from matches between top engines.

---

## 🔄 D4 Symmetry Augmentation

Following the data augmentation techniques detailed in UCLA's Gomoku AI study ([Chen, K.](https://www.physics.ucla.edu/~kevinchen/coding/gomoku.pdf)), the pipeline expands the training data **8-fold** by applying the complete dihedral D4 group transformations. For every board state and corresponding target policy, the following 8 variants are generated:
1. **Identity**: Original state.
2. **Rotation**: Rotate 90° clockwise.
3. **Rotation**: Rotate 180° clockwise.
4. **Rotation**: Rotate 270° clockwise.
5. **Horizontal Mirror**: Left-to-right flip.
6. **Vertical Mirror**: Top-to-bottom flip.
7. **Main Diagonal Mirror**: Transposition.
8. **Anti-Diagonal Mirror**: Anti-transposition.

These augmentations are defined in `src/augmentation.py` and are applied dynamically during training.

---

## 💻 Hardware & Software Requirements

- **GPU Acceleration**: Optimized for **Intel Arc B580** (Xe2 architecture) using **PyTorch XPU** (`torch==2.12.0+xpu` and `triton-xpu==3.7.1`).
- **Python Version**: `3.12`
- See [AGENTS.md](file:///c:/AAAAAAAAAAA_temp/desktop/Hephaestus_Repository/Colosseum/Gomoku_model/AGENTS.md) for detailed platform-specific configurations.

---

## 🚀 Getting Started

### 1. Installation
Install the necessary package dependencies in your virtual environment:
```bash
.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Verify the Codebase
Run the mock verification suite to test model shapes, loss functions, D4 transformations, and parser imports:
```bash
.venv\Scripts\python.exe C:\Users\Spade\.gemini\antigravity-ide\brain\738e0b97-6d78-49ff-ab34-3315eb202f1c\scratch\verify_pipeline.py
```

### 3. Launch Training
Run the main script to start training:
```bash
# Standard training
.venv\Scripts\python.exe train.py

# Training with student knowledge distillation (Stage 3)
.venv\Scripts\python.exe train.py --teacher checkpoints/best_teacher_model.pt

# Override hyperparameters
.venv\Scripts\python.exe train.py --epochs 100 --batch_size 128 --lr 5e-4
```
