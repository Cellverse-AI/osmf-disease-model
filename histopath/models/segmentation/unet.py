"""U-Net architecture components."""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Convolutional block with two convolutions, instance normalization, and ReLU.
    
    This is the basic building block used in U-Net architecture.
    Each block consists of:
        Conv2d -> InstanceNorm2d -> ReLU -> Conv2d -> InstanceNorm2d -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Attributes:
        conv1: First convolutional layer
        in1: First instance normalization layer
        conv2: Second convolutional layer
        in2: Second instance normalization layer
        relu: ReLU activation
    
    Example:
        >>> block = ConvBlock(in_channels=3, out_channels=64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = block(x)
        >>> print(out.shape)  # torch.Size([1, 64, 256, 256])
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ConvBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.in1 = nn.InstanceNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.in2 = nn.InstanceNorm2d(out_channels)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of convolutional block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        # First conv + norm + relu
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        
        # Second conv + norm + relu
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)
        
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net.
    
    Consists of a convolutional block followed by max pooling.
    Returns both the features (for skip connections) and the pooled output.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Attributes:
        conv: Convolutional block
        pool: Max pooling layer
    
    Example:
        >>> encoder = EncoderBlock(in_channels=3, out_channels=64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> features, pooled = encoder(x)
        >>> print(features.shape, pooled.shape)
        # torch.Size([1, 64, 256, 256]) torch.Size([1, 64, 128, 128])
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(EncoderBlock, self).__init__()
        
        # Convolutional block
        self.conv = ConvBlock(in_channels, out_channels)
        
        # Max pooling (2x2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of encoder block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Tuple of (features, pooled) where:
                - features: Output of conv block (B, C_out, H, W) for skip connection
                - pooled: Max pooled output (B, C_out, H/2, W/2)
        """
        # Apply convolutions
        features = self.conv(x)
        
        # Apply pooling
        pooled = self.pool(features)
        
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net.
    
    Consists of upsampling (transposed convolution), concatenation with
    skip connection from encoder, and a convolutional block.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Attributes:
        up: Transposed convolution for upsampling
        conv: Convolutional block
    
    Example:
        >>> decoder = DecoderBlock(in_channels=128, out_channels=64)
        >>> x = torch.randn(1, 128, 64, 64)
        >>> skip = torch.randn(1, 64, 128, 128)
        >>> out = decoder(x, skip)
        >>> print(out.shape)  # torch.Size([1, 64, 128, 128])
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DecoderBlock, self).__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2, 
            padding=0
        )
        
        # Convolutional block
        # Input channels = out_channels (from up) + out_channels (from skip)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of decoder block.
        
        Args:
            x: Input tensor from previous layer (B, C_in, H, W)
            skip: Skip connection from encoder (B, C_out, H*2, W*2)
        
        Returns:
            Output tensor of shape (B, C_out, H*2, W*2)
        """
        # Upsample
        x = self.up(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        
        return x


# Test function for development
def test_unet_components():
    """Test U-Net components with sample input."""
    print("Testing U-Net Components...")
    
    # Test ConvBlock
    print("\n1. Testing ConvBlock:")
    conv_block = ConvBlock(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 256, 256)
    out = conv_block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: (2, 64, 256, 256)")
    assert out.shape == (2, 64, 256, 256), "ConvBlock output shape mismatch"
    
    # Test EncoderBlock
    print("\n2. Testing EncoderBlock:")
    encoder = EncoderBlock(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 256, 256)
    features, pooled = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Features shape: {features.shape}")
    print(f"   Pooled shape: {pooled.shape}")
    print(f"   Expected features: (2, 64, 256, 256)")
    print(f"   Expected pooled: (2, 64, 128, 128)")
    assert features.shape == (2, 64, 256, 256), "EncoderBlock features shape mismatch"
    assert pooled.shape == (2, 64, 128, 128), "EncoderBlock pooled shape mismatch"
    
    # Test DecoderBlock
    print("\n3. Testing DecoderBlock:")
    decoder = DecoderBlock(in_channels=128, out_channels=64)
    x = torch.randn(2, 128, 64, 64)
    skip = torch.randn(2, 64, 128, 128)
    out = decoder(x, skip)
    print(f"   Input shape: {x.shape}")
    print(f"   Skip shape: {skip.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: (2, 64, 128, 128)")
    assert out.shape == (2, 64, 128, 128), "DecoderBlock output shape mismatch"
    
    print("\n✓ All U-Net component tests passed!")


if __name__ == "__main__":
    test_unet_components()
