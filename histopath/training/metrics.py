"""Evaluation metrics for segmentation."""

import torch
import numpy as np
from typing import Union, Optional


class Metrics:
    """
    Evaluation metrics for segmentation tasks.
    
    Provides static methods for computing common segmentation metrics:
    - Dice Coefficient (F1 Score)
    - IoU Score (Jaccard Index)
    - Pixel Accuracy
    - Precision
    - Recall
    
    All methods support both PyTorch tensors and NumPy arrays.
    """
    
    @staticmethod
    def dice_coefficient(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        smooth: float = 1e-6
    ) -> float:
        """
        Calculate Dice coefficient (F1 score).
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
            smooth: Smoothing factor to avoid division by zero
        
        Returns:
            Dice coefficient value in [0, 1]
        
        Example:
            >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
            >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
            >>> dice = Metrics.dice_coefficient(pred, target)
            >>> print(f"Dice: {dice:.4f}")
        """
        # Convert to torch tensor if numpy
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Apply sigmoid if values are not in [0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection
        intersection = (pred * target).sum()
        
        # Compute Dice coefficient
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return dice.item()
    
    @staticmethod
    def iou_score(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        smooth: float = 1e-6
    ) -> float:
        """
        Calculate IoU score (Jaccard index).
        
        IoU = |X ∩ Y| / |X ∪ Y|
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
            smooth: Smoothing factor to avoid division by zero
        
        Returns:
            IoU score value in [0, 1]
        
        Example:
            >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
            >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
            >>> iou = Metrics.iou_score(pred, target)
            >>> print(f"IoU: {iou:.4f}")
        """
        # Convert to torch tensor if numpy
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Apply sigmoid if values are not in [0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        # Compute IoU
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()
    
    @staticmethod
    def pixel_accuracy(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5
    ) -> float:
        """
        Calculate pixel-wise accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
        
        Returns:
            Pixel accuracy value in [0, 1]
        
        Example:
            >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
            >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
            >>> acc = Metrics.pixel_accuracy(pred, target)
            >>> print(f"Accuracy: {acc:.4f}")
        """
        # Convert to torch tensor if numpy
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Apply sigmoid if values are not in [0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute accuracy
        correct = (pred == target).sum()
        total = target.numel()
        accuracy = correct.float() / total
        
        return accuracy.item()
    
    @staticmethod
    def precision(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        smooth: float = 1e-6
    ) -> float:
        """
        Calculate precision.
        
        Precision = TP / (TP + FP)
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
            smooth: Smoothing factor to avoid division by zero
        
        Returns:
            Precision value in [0, 1]
        """
        # Convert to torch tensor if numpy
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Apply sigmoid if values are not in [0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute true positives and false positives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        
        # Compute precision
        precision = (tp + smooth) / (tp + fp + smooth)
        
        return precision.item()
    
    @staticmethod
    def recall(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        smooth: float = 1e-6
    ) -> float:
        """
        Calculate recall (sensitivity).
        
        Recall = TP / (TP + FN)
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
            smooth: Smoothing factor to avoid division by zero
        
        Returns:
            Recall value in [0, 1]
        """
        # Convert to torch tensor if numpy
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Apply sigmoid if values are not in [0, 1]
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute true positives and false negatives
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        
        # Compute recall
        recall = (tp + smooth) / (tp + fn + smooth)
        
        return recall.item()
    
    @staticmethod
    def compute_all_metrics(
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5
    ) -> dict:
        """
        Compute all metrics at once.
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for converting probabilities to binary
        
        Returns:
            Dictionary with all metric values
        
        Example:
            >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
            >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
            >>> metrics = Metrics.compute_all_metrics(pred, target)
            >>> for name, value in metrics.items():
            ...     print(f"{name}: {value:.4f}")
        """
        return {
            'dice': Metrics.dice_coefficient(pred, target, threshold),
            'iou': Metrics.iou_score(pred, target, threshold),
            'accuracy': Metrics.pixel_accuracy(pred, target, threshold),
            'precision': Metrics.precision(pred, target, threshold),
            'recall': Metrics.recall(pred, target, threshold)
        }


class MetricsTracker:
    """
    Track metrics over multiple batches.
    
    Accumulates metric values and computes averages.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     pred, target = model(batch), batch['mask']
        ...     tracker.update(pred, target)
        >>> avg_metrics = tracker.get_average()
        >>> print(avg_metrics)
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {
            'dice': [],
            'iou': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
    
    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5
    ):
        """
        Update metrics with new batch.
        
        Args:
            pred: Predicted probabilities or logits
            target: Ground truth binary masks
            threshold: Threshold for binary conversion
        """
        batch_metrics = Metrics.compute_all_metrics(pred, target, threshold)
        
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self) -> dict:
        """
        Get average metrics across all batches.
        
        Returns:
            Dictionary with average metric values
        """
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_current(self) -> dict:
        """
        Get most recent metric values.
        
        Returns:
            Dictionary with most recent metric values
        """
        current_metrics = {}
        for key, values in self.metrics.items():
            if values:
                current_metrics[key] = values[-1]
            else:
                current_metrics[key] = 0.0
        
        return current_metrics


# Test function for development
def test_metrics():
    """Test metrics functions."""
    print("Testing Metrics...")
    
    # Create sample data
    batch_size = 2
    pred = torch.sigmoid(torch.randn(batch_size, 1, 64, 64))
    target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
    
    # Test individual metrics
    print("\n1. Testing individual metrics:")
    dice = Metrics.dice_coefficient(pred, target)
    iou = Metrics.iou_score(pred, target)
    acc = Metrics.pixel_accuracy(pred, target)
    prec = Metrics.precision(pred, target)
    rec = Metrics.recall(pred, target)
    
    print(f"   Dice: {dice:.4f}")
    print(f"   IoU: {iou:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall: {rec:.4f}")
    
    assert 0 <= dice <= 1, "Dice should be in [0, 1]"
    assert 0 <= iou <= 1, "IoU should be in [0, 1]"
    assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
    
    # Test compute_all_metrics
    print("\n2. Testing compute_all_metrics:")
    all_metrics = Metrics.compute_all_metrics(pred, target)
    print(f"   All metrics: {all_metrics}")
    
    # Test MetricsTracker
    print("\n3. Testing MetricsTracker:")
    tracker = MetricsTracker()
    
    for i in range(3):
        pred_batch = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target_batch = torch.randint(0, 2, (2, 1, 64, 64)).float()
        tracker.update(pred_batch, target_batch)
    
    avg_metrics = tracker.get_average()
    print(f"   Average metrics: {avg_metrics}")
    
    current_metrics = tracker.get_current()
    print(f"   Current metrics: {current_metrics}")
    
    # Test with numpy arrays
    print("\n4. Testing with NumPy arrays:")
    pred_np = pred.numpy()
    target_np = target.numpy()
    dice_np = Metrics.dice_coefficient(pred_np, target_np)
    print(f"   Dice (NumPy): {dice_np:.4f}")
    assert abs(dice - dice_np) < 1e-5, "NumPy and PyTorch results should match"
    
    print("\n✓ All metrics tests passed!")


if __name__ == "__main__":
    test_metrics()
