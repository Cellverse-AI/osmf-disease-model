"""Test script to verify installation and basic functionality."""

import torch
import numpy as np

print("=" * 70)
print("Testing Histopath Package Installation")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    from histopath.config import Config
    from histopath.models.segmentation import CBAMUNet
    from histopath.data.loader import DataLoader
    from histopath.data.transforms import Transforms
    from histopath.training.losses import BCEDiceJaccardLoss
    from histopath.training.metrics import Metrics
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test configuration
print("\n2. Testing configuration...")
try:
    config = Config()
    assert config.get('data.batch_size') == 16
    print("   ✓ Configuration working")
except Exception as e:
    print(f"   ✗ Configuration error: {e}")
    exit(1)

# Test model
print("\n3. Testing model...")
try:
    model = CBAMUNet(in_channels=3, out_channels=1, base_channels=32)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1, 256, 256)
    print(f"   ✓ Model working (parameters: {model.get_num_parameters():,})")
except Exception as e:
    print(f"   ✗ Model error: {e}")
    exit(1)

# Test loss
print("\n4. Testing loss function...")
try:
    criterion = BCEDiceJaccardLoss(dice_weight=0.3, jaccard_weight=0.7)
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = criterion(pred, target)
    assert loss.item() >= 0
    print(f"   ✓ Loss function working (loss: {loss.item():.4f})")
except Exception as e:
    print(f"   ✗ Loss error: {e}")
    exit(1)

# Test metrics
print("\n5. Testing metrics...")
try:
    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    dice = Metrics.dice_coefficient(pred, target)
    iou = Metrics.iou_score(pred, target)
    assert 0 <= dice <= 1
    assert 0 <= iou <= 1
    print(f"   ✓ Metrics working (Dice: {dice:.4f}, IoU: {iou:.4f})")
except Exception as e:
    print(f"   ✗ Metrics error: {e}")
    exit(1)

# Test transforms
print("\n6. Testing transforms...")
try:
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    normalized = Transforms.normalize(img)
    tensor = Transforms.to_tensor(normalized)
    assert tensor.shape == (3, 256, 256)
    assert 0 <= tensor.max() <= 1
    print("   ✓ Transforms working")
except Exception as e:
    print(f"   ✗ Transform error: {e}")
    exit(1)

# Check CUDA availability
print("\n7. Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available (Device: {torch.cuda.get_device_name(0)})")
else:
    print("   ⚠ CUDA not available, will use CPU")

print("\n" + "=" * 70)
print("All tests passed! Installation successful.")
print("=" * 70)
print("\nYou can now:")
print("1. Train a model: python scripts/train_segmentation.py --help")
print("2. Run inference: python scripts/predict.py --help")
print("=" * 70)
