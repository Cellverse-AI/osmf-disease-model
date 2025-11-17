# Getting Started

## Installation Complete! ✓

Your histopathology analysis package is now installed and ready to use.

## What's Been Created

### Core Modules
- ✅ **Configuration Management** - YAML-based configuration
- ✅ **Data Loading** - Image and mask loading with preprocessing
- ✅ **CBAM U-Net Model** - Complete segmentation architecture
- ✅ **Training Pipeline** - Full training loop with checkpointing
- ✅ **Inference Engine** - Batch prediction on images
- ✅ **Loss Functions** - BCE-Dice-Jaccard combined loss
- ✅ **Metrics** - Dice, IoU, accuracy, precision, recall
- ✅ **Utilities** - Visualization, logging, I/O

### Scripts
- ✅ **train_segmentation.py** - Train models
- ✅ **predict.py** - Run inference

## Quick Test

Run the test script to verify everything works:
```bash
python test_installation.py
```

## Training Your Model

### 1. Prepare Your Data

Organize your data like this:
```
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

### 2. Train

```bash
python scripts/train_segmentation.py \
    --train-images "Hist joint dataset/Untitled Folder/tissue images" \
    --train-masks "Hist joint dataset/mask binary" \
    --epochs 50 \
    --batch-size 16 \
    --output-dir "outputs"
```

### 3. Monitor Training

Training will:
- Show progress bars for each epoch
- Display loss and metrics (Dice, IoU)
- Save checkpoints every 5 epochs
- Save best model based on validation loss
- Apply early stopping if no improvement

### 4. Run Inference

```bash
python scripts/predict.py \
    --checkpoint "outputs/checkpoints/best_model.pth" \
    --input-dir "path/to/test/images" \
    --output-dir "predictions" \
    --threshold 0.5
```

## Configuration

Create a custom config file (optional):

```yaml
# my_config.yaml
data:
  image_size: [256, 256]
  batch_size: 16

training:
  epochs: 50
  learning_rate: 0.0001
  loss_weights:
    dice: 0.3
    jaccard: 0.7
```

Use it:
```bash
python scripts/train_segmentation.py --config my_config.yaml ...
```

## Model Architecture

**CBAM U-Net** (7.7M parameters):
- Encoder: 4 blocks (32 → 64 → 128 → 256 channels)
- Bottleneck: 512 channels
- Decoder: 4 blocks with CBAM attention on skip connections
- Output: Binary segmentation mask

## Loss Function

Combined loss (from your notebook):
- 30% Dice Loss
- 70% Jaccard Loss

## Next Steps

1. **Train on your data**: Use the training script with your dataset
2. **Evaluate**: Check metrics and visualizations
3. **Tune**: Adjust hyperparameters in config
4. **Predict**: Run inference on new images

## Troubleshooting

### CUDA not available
- The code will automatically use CPU if CUDA is not available
- Training will be slower but will work

### Out of memory
- Reduce batch size: `--batch-size 8`
- Reduce image size in config

### Poor results
- Train for more epochs
- Adjust loss weights
- Check data quality and preprocessing

## File Structure

```
histopath/              # Main package
├── config/            # Configuration
├── data/              # Data loading
├── models/            # Model architectures
├── training/          # Training pipeline
├── inference/         # Inference engine
└── utils/             # Utilities

scripts/               # Executable scripts
├── train_segmentation.py
└── predict.py

outputs/               # Training outputs
├── checkpoints/       # Model checkpoints
└── logs/              # Training logs
```

## Support

The code is based on your original Jupyter notebooks and has been refactored for:
- Better organization
- Reusability
- Command-line interface
- Proper error handling
- Documentation

All original functionality is preserved!
