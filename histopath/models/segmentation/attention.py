"""Attention mechanisms for segmentation models."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Applies attention across channels using both average and max pooling,
    followed by a shared MLP to generate channel-wise attention weights.
    
    This is part of the CBAM (Convolutional Block Attention Module) architecture.
    
    Args:
        in_planes (int): Number of input channels
        ratio (int): Reduction ratio for the MLP. Default: 16
    
    Attributes:
        avg_pool: Adaptive average pooling layer
        max_pool: Adaptive max pooling layer
        fc: Shared MLP (implemented as 1x1 convolutions)
        sigmoid: Sigmoid activation for attention weights
    
    Example:
        >>> ca = ChannelAttention(in_planes=64, ratio=16)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> attention = ca(x)
        >>> print(attention.shape)  # torch.Size([1, 64, 1, 1])
    """
    
    def __init__(self, in_planes: int, ratio: int = 16):
        """
        Initialize Channel Attention module.
        
        Args:
            in_planes: Number of input channels
            ratio: Reduction ratio for MLP
        """
        super(ChannelAttention, self).__init__()
        
        # Global pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP (implemented as 1x1 convolutions)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
        # Sigmoid for attention weights
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Applies attention across spatial dimensions by aggregating channel information
    using both average and max pooling, followed by a convolution layer.
    
    This is part of the CBAM (Convolutional Block Attention Module) architecture.
    
    Args:
        kernel_size (int): Kernel size for the convolution. Default: 7
    
    Attributes:
        conv1: Convolution layer to generate spatial attention
        sigmoid: Sigmoid activation for attention weights
    
    Example:
        >>> sa = SpatialAttention(kernel_size=7)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> attention = sa(x)
        >>> print(attention.shape)  # torch.Size([1, 1, 32, 32])
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention module.
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super(SpatialAttention, self).__init__()
        
        # Convolution to generate spatial attention
        # Input: 2 channels (avg + max), Output: 1 channel
        self.conv1 = nn.Conv2d(
            2, 1, 
            kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        
        # Sigmoid for attention weights
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Spatial attention weights of shape (B, 1, H, W)
        """
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv1(x_cat))
        
        return attention


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention sequentially.
    First applies channel attention, then spatial attention.
    
    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    
    Args:
        in_planes (int): Number of input channels
        ratio (int): Reduction ratio for channel attention. Default: 16
        kernel_size (int): Kernel size for spatial attention. Default: 7
    
    Attributes:
        ca: Channel attention module
        sa: Spatial attention module
    
    Example:
        >>> cbam = CBAMBlock(in_planes=64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = cbam(x)
        >>> print(out.shape)  # torch.Size([1, 64, 32, 32])
    """
    
    def __init__(
        self, 
        in_planes: int, 
        ratio: int = 16, 
        kernel_size: int = 7
    ):
        """
        Initialize CBAM block.
        
        Args:
            in_planes: Number of input channels
            ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAMBlock, self).__init__()
        
        # Channel attention
        self.ca = ChannelAttention(in_planes, ratio)
        
        # Spatial attention
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM.
        
        Applies channel attention followed by spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-refined features of shape (B, C, H, W)
        """
        # Apply channel attention
        x_out = self.ca(x) * x
        
        # Apply spatial attention
        x_out = self.sa(x_out) * x_out
        
        return x_out


# Test function for development
def test_attention_modules():
    """Test attention modules with sample input."""
    print("Testing Attention Modules...")
    
    # Test input
    batch_size = 2
    channels = 64
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    # Test Channel Attention
    print("\n1. Testing Channel Attention:")
    ca = ChannelAttention(in_planes=channels, ratio=16)
    ca_out = ca(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {ca_out.shape}")
    print(f"   Expected: ({batch_size}, {channels}, 1, 1)")
    assert ca_out.shape == (batch_size, channels, 1, 1), "Channel attention output shape mismatch"
    
    # Test Spatial Attention
    print("\n2. Testing Spatial Attention:")
    sa = SpatialAttention(kernel_size=7)
    sa_out = sa(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {sa_out.shape}")
    print(f"   Expected: ({batch_size}, 1, {height}, {width})")
    assert sa_out.shape == (batch_size, 1, height, width), "Spatial attention output shape mismatch"
    
    # Test CBAM Block
    print("\n3. Testing CBAM Block:")
    cbam = CBAMBlock(in_planes=channels, ratio=16, kernel_size=7)
    cbam_out = cbam(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {cbam_out.shape}")
    print(f"   Expected: ({batch_size}, {channels}, {height}, {width})")
    assert cbam_out.shape == x.shape, "CBAM output shape mismatch"
    
    print("\n✓ All attention module tests passed!")


if __name__ == "__main__":
    test_attention_modules()
