"""CBAM U-Net model for histopathological image segmentation."""

import torch
import torch.nn as nn
from typing import Optional

from histopath.models.segmentation.unet import ConvBlock, EncoderBlock, DecoderBlock
from histopath.models.segmentation.attention import CBAMBlock


class CBAMUNet(nn.Module):
    """
    U-Net with CBAM (Convolutional Block Attention Module) on skip connections.
    
    This architecture combines the U-Net segmentation network with CBAM attention
    mechanisms applied to the skip connections, allowing the model to focus on
    relevant features during the decoding process.
    
    Architecture:
        - Encoder: 4 encoder blocks with increasing channels (32, 64, 128, 256)
        - Bottleneck: Convolutional block with 512 channels
        - Decoder: 4 decoder blocks with decreasing channels (256, 128, 64, 32)
        - CBAM: Applied to each skip connection before concatenation
        - Output: 1x1 convolution for final segmentation mask
    
    Args:
        in_channels (int): Number of input channels. Default: 3 (RGB)
        out_channels (int): Number of output channels. Default: 1 (binary segmentation)
        base_channels (int): Base number of channels. Default: 32
        depth (int): Depth of U-Net (number of encoder/decoder blocks). Default: 4
    
    Attributes:
        e1, e2, e3, e4: Encoder blocks
        bottleneck: Bottleneck convolutional block
        d4, d3, d2, d1: Decoder blocks
        cbam1, cbam2, cbam3, cbam4: CBAM attention blocks for skip connections
        outputs: Final 1x1 convolution for output
    
    Example:
        >>> model = CBAMUNet(in_channels=3, out_channels=1)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4
    ):
        """
        Initialize CBAM U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            base_channels: Base number of channels in first layer
            depth: Depth of U-Net (currently fixed at 4)
        """
        super(CBAMUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Calculate channel sizes for each level
        c1 = base_channels      # 32
        c2 = base_channels * 2  # 64
        c3 = base_channels * 4  # 128
        c4 = base_channels * 8  # 256
        c5 = base_channels * 16 # 512
        
        # Encoder blocks
        self.e1 = EncoderBlock(in_channels, c1)
        self.e2 = EncoderBlock(c1, c2)
        self.e3 = EncoderBlock(c2, c3)
        self.e4 = EncoderBlock(c3, c4)
        
        # Bottleneck
        self.bottleneck = ConvBlock(c4, c5)
        
        # Decoder blocks
        self.d4 = DecoderBlock(c5, c4)
        self.d3 = DecoderBlock(c4, c3)
        self.d2 = DecoderBlock(c3, c2)
        self.d1 = DecoderBlock(c2, c1)
        
        # CBAM attention blocks for skip connections
        self.cbam1 = CBAMBlock(c1)
        self.cbam2 = CBAMBlock(c2)
        self.cbam3 = CBAMBlock(c3)
        self.cbam4 = CBAMBlock(c4)
        
        # Final output layer
        self.outputs = nn.Conv2d(c1, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM U-Net.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Output segmentation mask of shape (B, C_out, H, W)
        """
        # Encoder path
        s1, p1 = self.e1(x)    # Skip1, Pooled1
        s2, p2 = self.e2(p1)   # Skip2, Pooled2
        s3, p3 = self.e3(p2)   # Skip3, Pooled3
        s4, p4 = self.e4(p3)   # Skip4, Pooled4
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoder path with CBAM attention on skip connections
        d4 = self.d4(bottleneck, self.cbam4(s4))
        d3 = self.d3(d4, self.cbam3(s3))
        d2 = self.d2(d3, self.cbam2(s2))
        d1 = self.d1(d2, self.cbam1(s1))
        
        # Final output
        outputs = self.outputs(d1)
        
        return outputs
    
    def load_weights(self, path: str, device: Optional[torch.device] = None) -> None:
        """
        Load model weights from file.
        
        Args:
            path: Path to weights file (.pth)
            device: Device to load weights to (None for automatic)
        
        Example:
            >>> model = CBAMUNet()
            >>> model.load_weights("checkpoints/best_model.pth")
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        
        print(f"Loaded model weights from {path}")
    
    def save_weights(self, path: str, **kwargs) -> None:
        """
        Save model weights to file.
        
        Args:
            path: Path to save weights (.pth)
            **kwargs: Additional information to save (e.g., epoch, optimizer state)
        
        Example:
            >>> model = CBAMUNet()
            >>> model.save_weights("checkpoints/model.pth", epoch=10)
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict along with any additional info
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'base_channels': self.base_channels,
                'depth': self.depth
            },
            **kwargs
        }
        
        torch.save(checkpoint, path)
        print(f"Saved model weights to {path}")
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        
        Example:
            >>> model = CBAMUNet()
            >>> print(f"Parameters: {model.get_num_parameters():,}")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> None:
        """
        Print model summary.
        
        Example:
            >>> model = CBAMUNet()
            >>> model.summary()
        """
        print("=" * 70)
        print("CBAM U-Net Model Summary")
        print("=" * 70)
        print(f"Input channels:  {self.in_channels}")
        print(f"Output channels: {self.out_channels}")
        print(f"Base channels:   {self.base_channels}")
        print(f"Depth:           {self.depth}")
        print(f"Total parameters: {self.get_num_parameters():,}")
        print("=" * 70)
        
        # Print layer structure
        print("\nArchitecture:")
        print("  Encoder:")
        print(f"    E1: {self.in_channels} -> {self.base_channels}")
        print(f"    E2: {self.base_channels} -> {self.base_channels * 2}")
        print(f"    E3: {self.base_channels * 2} -> {self.base_channels * 4}")
        print(f"    E4: {self.base_channels * 4} -> {self.base_channels * 8}")
        print(f"  Bottleneck: {self.base_channels * 8} -> {self.base_channels * 16}")
        print("  Decoder:")
        print(f"    D4: {self.base_channels * 16} -> {self.base_channels * 8}")
        print(f"    D3: {self.base_channels * 8} -> {self.base_channels * 4}")
        print(f"    D2: {self.base_channels * 4} -> {self.base_channels * 2}")
        print(f"    D1: {self.base_channels * 2} -> {self.base_channels}")
        print(f"  Output: {self.base_channels} -> {self.out_channels}")
        print("=" * 70)


# Test function for development
def test_cbam_unet():
    """Test CBAM U-Net model."""
    print("Testing CBAM U-Net Model...")
    
    # Create model
    model = CBAMUNet(in_channels=3, out_channels=1, base_channels=32)
    
    # Print summary
    model.summary()
    
    # Test forward pass
    print("\nTesting forward pass:")
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 1, 256, 256)")
    
    assert output.shape == (batch_size, 1, 256, 256), "Output shape mismatch"
    
    # Test save/load
    print("\nTesting save/load:")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")
        model.save_weights(save_path, epoch=1, loss=0.5)
        
        # Create new model and load weights
        model2 = CBAMUNet(in_channels=3, out_channels=1, base_channels=32)
        model2.load_weights(save_path)
        
        # Verify outputs match
        with torch.no_grad():
            output2 = model2(x)
        
        assert torch.allclose(output, output2), "Loaded model outputs don't match"
        print("✓ Save/load test passed")
    
    print("\n✓ All CBAM U-Net tests passed!")


if __name__ == "__main__":
    test_cbam_unet()
