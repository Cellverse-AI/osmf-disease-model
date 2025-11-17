"""Image transformation and preprocessing utilities."""

import numpy as np
import torch
from typing import Callable, Tuple, Optional, Any


class Transforms:
    """
    Image transformation and augmentation utilities.
    
    Provides preprocessing functions for histopathological images
    including normalization, resizing, and tensor conversion.
    """
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Normalized image in [0, 1] range
        
        Example:
            >>> img = np.array([0, 128, 255], dtype=np.uint8)
            >>> normalized = Transforms.normalize(img)
            >>> print(normalized)  # [0., 0.5, 1.]
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            return image.astype(np.float32) / 255.0
        else:
            return image.astype(np.float32)
    
    @staticmethod
    def denormalize(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255] range.
        
        Args:
            image: Normalized image in [0, 1] range
        
        Returns:
            Image in [0, 255] range as uint8
        """
        return (image * 255).astype(np.uint8)
    
    @staticmethod
    def to_tensor(image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        
        Converts from (H, W, C) to (C, H, W) format for PyTorch.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
        
        Returns:
            PyTorch tensor in (C, H, W) format
        
        Example:
            >>> img = np.random.rand(256, 256, 3)
            >>> tensor = Transforms.to_tensor(img)
            >>> print(tensor.shape)  # torch.Size([3, 256, 256])
        """
        # Handle grayscale images
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Convert from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        
        return tensor.float()
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.
        
        Converts from (C, H, W) to (H, W, C) format.
        
        Args:
            tensor: PyTorch tensor in (C, H, W) format
        
        Returns:
            Numpy array in (H, W, C) format
        """
        # Move to CPU and convert to numpy
        array = tensor.cpu().numpy()
        
        # Convert from (C, H, W) to (H, W, C)
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)
        
        # Squeeze single channel dimension
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
        
        return array
    
    @staticmethod
    def get_train_transforms(config: Any) -> Callable:
        """
        Get training transformation pipeline.
        
        Args:
            config: Configuration object
        
        Returns:
            Callable transformation function
        
        Example:
            >>> from histopath.config import Config
            >>> config = Config()
            >>> transform = Transforms.get_train_transforms(config)
            >>> img_tensor = transform(image)
        """
        def transform(image: np.ndarray) -> torch.Tensor:
            """Apply training transformations."""
            # Normalize if configured
            if config.get("data.normalize", True):
                image = Transforms.normalize(image)
            
            # Convert to tensor
            tensor = Transforms.to_tensor(image)
            
            return tensor
        
        return transform
    
    @staticmethod
    def get_val_transforms(config: Any) -> Callable:
        """
        Get validation transformation pipeline.
        
        Args:
            config: Configuration object
        
        Returns:
            Callable transformation function
        """
        def transform(image: np.ndarray) -> torch.Tensor:
            """Apply validation transformations."""
            # Normalize if configured
            if config.get("data.normalize", True):
                image = Transforms.normalize(image)
            
            # Convert to tensor
            tensor = Transforms.to_tensor(image)
            
            return tensor
        
        return transform
    
    @staticmethod
    def prepare_mask(mask: np.ndarray) -> torch.Tensor:
        """
        Prepare mask for training.
        
        Converts mask to binary format and appropriate tensor shape.
        
        Args:
            mask: Input mask as numpy array
        
        Returns:
            PyTorch tensor of shape (1, H, W)
        
        Example:
            >>> mask = np.array([[0, 255], [128, 255]], dtype=np.uint8)
            >>> mask_tensor = Transforms.prepare_mask(mask)
            >>> print(mask_tensor.shape)  # torch.Size([1, 2, 2])
        """
        # Normalize mask to [0, 1]
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        elif mask.max() > 1.0:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)
        
        # Ensure 2D
        if mask.ndim == 3:
            # Take first channel if multi-channel
            mask = mask[:, :, 0]
        
        # Add channel dimension and convert to tensor
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        tensor = torch.from_numpy(mask).float()
        
        return tensor
    
    @staticmethod
    def apply_threshold(
        prediction: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Apply threshold to prediction to get binary mask.
        
        Args:
            prediction: Prediction array with values in [0, 1]
            threshold: Threshold value
        
        Returns:
            Binary mask (0 or 1)
        """
        return (prediction > threshold).astype(np.uint8)
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (height, width)
        
        Returns:
            Resized image
        """
        from PIL import Image as PILImage
        
        # Convert to PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        pil_img = PILImage.fromarray(image)
        
        # Resize (PIL uses width, height)
        resized = pil_img.resize((target_size[1], target_size[0]))
        
        return np.array(resized)


class Compose:
    """
    Compose multiple transforms together.
    
    Example:
        >>> transforms = Compose([
        ...     Transforms.normalize,
        ...     Transforms.to_tensor
        ... ])
        >>> output = transforms(image)
    """
    
    def __init__(self, transforms: list):
        """
        Initialize composed transforms.
        
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> Any:
        """
        Apply all transforms sequentially.
        
        Args:
            image: Input image
        
        Returns:
            Transformed image
        """
        for transform in self.transforms:
            image = transform(image)
        return image
