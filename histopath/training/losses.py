"""Loss functions for segmentation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCEDiceJaccardLoss(nn.Module):
    """
    Combined BCE, Dice, and Jaccard loss for segmentation.
    
    This loss function combines three complementary losses:
    - Binary Cross-Entropy (BCE): Pixel-wise classification loss
    - Dice Loss: Overlap-based loss focusing on region similarity
    - Jaccard Loss (IoU): Intersection over Union loss
    
    The final loss is a weighted combination:
        loss = bce_weight * BCE + dice_weight * Dice + jaccard_weight * Jaccard
    
    Args:
        bce_weight (float): Weight for BCE loss. Default: 0.0
        dice_weight (float): Weight for Dice loss. Default: 0.3
        jaccard_weight (float): Weight for Jaccard loss. Default: 0.7
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0
    
    Attributes:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        jaccard_weight: Weight for Jaccard loss
        smooth: Smoothing factor
        bce: BCE with logits loss module
    
    Example:
        >>> criterion = BCEDiceJaccardLoss(dice_weight=0.3, jaccard_weight=0.7)
        >>> pred = torch.randn(2, 1, 256, 256)  # Logits
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = criterion(pred, target)
        >>> print(loss.item())
    """
    
    def __init__(
        self,
        bce_weight: float = 0.0,
        dice_weight: float = 0.3,
        jaccard_weight: float = 0.7,
        smooth: float = 1.0
    ):
        """
        Initialize combined loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            jaccard_weight: Weight for Jaccard loss
            smooth: Smoothing factor
        """
        super(BCEDiceJaccardLoss, self).__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth
        
        # BCE with logits (more numerically stable)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits of shape (B, 1, H, W)
            target: Ground truth masks of shape (B, 1, H, W) with values in [0, 1]
        
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Binary Cross-Entropy loss
        if self.bce_weight > 0:
            bce_loss = self.bce(pred, target)
            total_loss += self.bce_weight * bce_loss
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice_loss
        
        # Jaccard loss
        if self.jaccard_weight > 0:
            jaccard_loss = self.jaccard_loss(pred, target)
            total_loss += self.jaccard_weight * jaccard_loss
        
        return total_loss
    
    def dice_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
        Dice loss = 1 - Dice coefficient
        
        Args:
            pred: Predicted logits of shape (B, 1, H, W)
            target: Ground truth masks of shape (B, 1, H, W)
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred * target).sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Dice loss
        return 1 - dice
    
    def jaccard_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jaccard loss (IoU loss).
        
        Jaccard index = |X ∩ Y| / |X ∪ Y|
        Jaccard loss = 1 - Jaccard index
        
        Args:
            pred: Predicted logits of shape (B, 1, H, W)
            target: Ground truth masks of shape (B, 1, H, W)
        
        Returns:
            Jaccard loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Apply threshold to get binary predictions
        pred = (pred > 0.5).float()
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection
        
        # Jaccard index
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        # Jaccard loss
        return 1 - jaccard


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    
    Standalone Dice loss implementation.
    
    Args:
        smooth (float): Smoothing factor. Default: 1.0
    
    Example:
        >>> criterion = DiceLoss()
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = criterion(pred, target)
    """
    
    def __init__(self, smooth: float = 1.0):
        """Initialize Dice loss."""
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth masks
        
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class JaccardLoss(nn.Module):
    """
    Jaccard loss (IoU loss) for segmentation.
    
    Standalone Jaccard loss implementation.
    
    Args:
        smooth (float): Smoothing factor. Default: 1.0
        threshold (float): Threshold for binary predictions. Default: 0.5
    
    Example:
        >>> criterion = JaccardLoss()
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = criterion(pred, target)
    """
    
    def __init__(self, smooth: float = 1.0, threshold: float = 0.5):
        """Initialize Jaccard loss."""
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jaccard loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth masks
        
        Returns:
            Jaccard loss value
        """
        pred = torch.sigmoid(pred)
        pred = (pred > self.threshold).float()
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Functional Dice loss.
    
    Standalone function version of Dice loss for convenience.
    
    Args:
        pred: Predicted logits of shape (B, C, H, W)
        target: Ground truth masks of shape (B, C, H, W)
        smooth: Smoothing factor
    
    Returns:
        Dice loss value
    
    Example:
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = dice_loss(pred, target)
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def jaccard_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    smooth: float = 1.0,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Functional Jaccard loss.
    
    Standalone function version of Jaccard loss for convenience.
    
    Args:
        pred: Predicted logits of shape (B, C, H, W)
        target: Ground truth masks of shape (B, C, H, W)
        smooth: Smoothing factor
        threshold: Threshold for binary predictions
    
    Returns:
        Jaccard loss value
    
    Example:
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = jaccard_loss(pred, target)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    
    jaccard = (intersection + smooth) / (union + smooth)
    
    return 1 - jaccard


# Test function for development
def test_losses():
    """Test loss functions."""
    print("Testing Loss Functions...")
    
    # Create sample data
    batch_size = 2
    pred = torch.randn(batch_size, 1, 64, 64)  # Logits
    target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
    
    # Test BCEDiceJaccardLoss
    print("\n1. Testing BCEDiceJaccardLoss:")
    criterion = BCEDiceJaccardLoss(bce_weight=0.0, dice_weight=0.3, jaccard_weight=0.7)
    loss = criterion(pred, target)
    print(f"   Loss value: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test DiceLoss
    print("\n2. Testing DiceLoss:")
    dice_criterion = DiceLoss()
    dice_loss_val = dice_criterion(pred, target)
    print(f"   Dice loss: {dice_loss_val.item():.4f}")
    assert dice_loss_val.item() >= 0, "Dice loss should be non-negative"
    
    # Test JaccardLoss
    print("\n3. Testing JaccardLoss:")
    jaccard_criterion = JaccardLoss()
    jaccard_loss_val = jaccard_criterion(pred, target)
    print(f"   Jaccard loss: {jaccard_loss_val.item():.4f}")
    assert jaccard_loss_val.item() >= 0, "Jaccard loss should be non-negative"
    
    # Test functional versions
    print("\n4. Testing functional losses:")
    func_dice = dice_loss(pred, target)
    func_jaccard = jaccard_loss(pred, target)
    print(f"   Functional Dice loss: {func_dice.item():.4f}")
    print(f"   Functional Jaccard loss: {func_jaccard.item():.4f}")
    
    # Test backward pass
    print("\n5. Testing backward pass:")
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None, "Gradients should be computed"
    print("   ✓ Backward pass successful")
    
    print("\n✓ All loss function tests passed!")


if __name__ == "__main__":
    test_losses()
