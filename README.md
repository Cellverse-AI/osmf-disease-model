# Histopathology Image Analysis

A Python package for histopathological image analysis including cell density segmentation using CBAM U-Net and fibrosis classification.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Cell Density Segmentation**: CBAM U-Net model for semantic segmentation of histopathological images
- **Training Pipeline**: Complete training infrastructure with:
  - Checkpointing and model saving
  - Early stopping
  - Progress tracking with tqdm
  - Comprehensive metrics logging
- **Inference Engine**: Batch prediction on images with post-processing
- **Evaluation Metrics**: Dice coefficient, IoU, pixel accuracy, precision, recall
- **Visualization**: Training curves and segmentation results
- **Modular Architecture**: Clean, well-documented, and extensible codebase

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/histopath-analysis.git
cd histopath-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. Verify installation:
```bash
python test_installation.py
```

## Quick Start

### Training

Train a segmentation model:

```bash
python scripts/train_segmentation.py \
    --train-images "path/to/train/images" \
    --train-masks "path/to/train/masks" \
    --val-images "path/to/val/images" \
    --val-masks "path/to/val/masks" \
    --epochs 50 \
    --batch-size 16 \
    --output-dir "outputs"
```

### Inference

Run inference on new images:

```bash
python scripts/predict.py \
    --checkpoint "outputs/checkpoints/best_model.pth" \
    --input-dir "path/to/test/images" \
    --output-dir "path/to/predictions" \
    --threshold 0.5
```

## Project Structure

```
histopath/
├── config/          # Configuration management
├── data/            # Data loading and preprocessing
├── models/          # Model architectures
│   └── segmentation/  # CBAM U-Net
├── training/        # Training pipeline
├── inference/       # Inference engine
├── utils/           # Utilities
└── analysis/        # Analysis tools

scripts/
├── train_segmentation.py  # Training script
└── predict.py             # Inference script
```

## Configuration

Create a YAML configuration file or use the defaults:

```yaml
data:
  image_size: [256, 256]
  batch_size: 16
  num_workers: 4

model:
  in_channels: 3
  out_channels: 1
  base_channels: 32

training:
  epochs: 50
  learning_rate: 0.0001
  loss_weights:
    dice: 0.3
    jaccard: 0.7
```

## Model Architecture

The CBAM U-Net combines:
- U-Net encoder-decoder architecture
- CBAM (Convolutional Block Attention Module) on skip connections
- Instance normalization
- Combined BCE-Dice-Jaccard loss

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

## License

MIT License


## Model Architecture

### CBAM U-Net

The segmentation model uses a U-Net architecture enhanced with Convolutional Block Attention Module (CBAM):

- **Encoder**: 4 blocks with increasing channels (32 → 64 → 128 → 256)
- **Bottleneck**: 512 channels
- **Decoder**: 4 blocks with CBAM attention on skip connections
- **Total Parameters**: ~7.7M
- **Loss Function**: Combined Dice (30%) + Jaccard (70%)

## Configuration

Create a custom configuration file (optional):

```yaml
# config.yaml
data:
  image_size: [256, 256]
  batch_size: 16
  num_workers: 4

model:
  in_channels: 3
  out_channels: 1
  base_channels: 32

training:
  epochs: 50
  learning_rate: 0.0001
  loss_weights:
    dice: 0.3
    jaccard: 0.7
```

Use it with:
```bash
python scripts/train_segmentation.py --config config.yaml ...
```

## Project Structure

```
histopath/
├── config/          # Configuration management
├── data/            # Data loading and preprocessing
├── models/          # Model architectures
│   └── segmentation/  # CBAM U-Net
├── training/        # Training pipeline
├── inference/       # Inference engine
├── utils/           # Utilities (logging, visualization, I/O)
└── analysis/        # Analysis tools

scripts/
├── train_segmentation.py  # Training script
└── predict.py             # Inference script
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- pillow
- scikit-image
- opencv-python-headless
- pandas
- matplotlib
- seaborn
- tqdm
- pyyaml
- scipy

See `requirements.txt` for complete list.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

