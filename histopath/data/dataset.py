"""PyTorch Dataset classes for histopathological images."""

import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
import numpy as np

from histopath.data.loader import DataLoader
from histopath.data.transforms import Transforms


class HistopathDataset(Dataset):
    """
    PyTorch Dataset for histopathological images.
    
    Supports loading image-mask pairs for segmentation tasks.
    Can also be used for inference (images only, no masks).
    
    Attributes:
        image_dir (str): Directory containing images
        mask_dir (Optional[str]): Directory containing masks (None for inference)
        transform (Optional[Callable]): Transform to apply to images
        target_size (Tuple[int, int]): Target size for images
        image_files (List[str]): List of image filenames
        mask_files (Optional[List[str]]): List of mask filenames
    
    Example:
        >>> dataset = HistopathDataset(
        ...     image_dir="data/train/images",
        ...     mask_dir="data/train/masks",
        ...     target_size=(256, 256)
        ... )
        >>> image, mask = dataset[0]
        >>> print(image.shape, mask.shape)
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks (None for inference mode)
            transform: Optional transform to apply to images
            target_size: Target size (height, width) for images
            normalize: Whether to normalize images to [0, 1]
        
        Raises:
            FileNotFoundError: If image_dir doesn't exist
            ValueError: If no images found or image-mask mismatch
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        
        # Verify image directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Get image filenames
        self.image_files = DataLoader.get_image_filenames(image_dir)
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        # Get mask filenames if mask_dir provided
        if mask_dir is not None:
            if not os.path.exists(mask_dir):
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
            
            # Verify correspondence between images and masks
            self.image_files, self.mask_files = DataLoader.verify_image_mask_correspondence(
                image_dir, mask_dir
            )
        else:
            self.mask_files = None
        
        print(f"Initialized dataset with {len(self.image_files)} images")
    
    def __len__(self) -> int:
        """
        Get dataset size.
        
        Returns:
            Number of images in dataset
        """
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get image and mask pair.
        
        Args:
            idx: Index of item to retrieve
        
        Returns:
            Tuple of (image_tensor, mask_tensor) where:
                - image_tensor: shape (C, H, W)
                - mask_tensor: shape (1, H, W) or None if no masks
        
        Example:
            >>> dataset = HistopathDataset("data/images", "data/masks")
            >>> image, mask = dataset[0]
            >>> print(image.shape, mask.shape)
        """
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = DataLoader.load_image(image_path, self.target_size)
        
        # Load mask if available
        mask = None
        if self.mask_files is not None:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = DataLoader.load_image(mask_path, self.target_size)
        
        # Apply normalization
        if self.normalize:
            image = Transforms.normalize(image)
            if mask is not None:
                mask = Transforms.normalize(mask)
        
        # Apply custom transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = Transforms.to_tensor(image)
        
        # Prepare mask
        if mask is not None:
            mask = Transforms.prepare_mask(mask)
        
        return image, mask
    
    def get_image_path(self, idx: int) -> str:
        """
        Get path to image at index.
        
        Args:
            idx: Index of image
        
        Returns:
            Full path to image file
        """
        return os.path.join(self.image_dir, self.image_files[idx])
    
    def get_mask_path(self, idx: int) -> Optional[str]:
        """
        Get path to mask at index.
        
        Args:
            idx: Index of mask
        
        Returns:
            Full path to mask file or None if no masks
        """
        if self.mask_files is None:
            return None
        return os.path.join(self.mask_dir, self.mask_files[idx])


class PatchDataset(Dataset):
    """
    PyTorch Dataset for pre-loaded image patches.
    
    This class replicates the behavior from the original notebook
    where images and masks are pre-loaded into memory as numpy arrays.
    
    Attributes:
        images (np.ndarray): Array of images (N, H, W, C)
        masks (Optional[np.ndarray]): Array of masks (N, H, W)
        transform (Optional[Callable]): Transform to apply
    
    Example:
        >>> images = np.random.rand(100, 256, 256, 3)
        >>> masks = np.random.rand(100, 256, 256)
        >>> dataset = PatchDataset(images, masks)
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Initialize patch dataset.
        
        Args:
            images: Array of images (N, H, W, C)
            masks: Optional array of masks (N, H, W)
            transform: Optional transform to apply
            normalize: Whether to normalize to [0, 1]
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        self.normalize = normalize
        
        # Validate shapes
        if masks is not None:
            assert len(images) == len(masks), \
                f"Number of images ({len(images)}) must match number of masks ({len(masks)})"
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get image and mask pair.
        
        Args:
            idx: Index of item
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Get image and mask
        image = self.images[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32) if self.masks is not None else None
        
        # Normalize if needed
        if self.normalize:
            image = Transforms.normalize(image)
            if mask is not None:
                mask = Transforms.normalize(mask)
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = Transforms.to_tensor(image)
        
        # Prepare mask
        if mask is not None:
            mask = Transforms.prepare_mask(mask)
        
        return image, mask


def create_dataloaders(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: Optional[str] = None,
    val_mask_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    shuffle_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_image_dir: Training images directory
        train_mask_dir: Training masks directory
        val_image_dir: Validation images directory (optional)
        val_mask_dir: Validation masks directory (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        target_size: Target image size
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader)
        val_loader is None if validation directories not provided
    
    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     "data/train/images",
        ...     "data/train/masks",
        ...     batch_size=16
        ... )
    """
    # Create training dataset
    train_dataset = HistopathDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        target_size=target_size
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create validation dataset if directories provided
    val_loader = None
    if val_image_dir is not None and val_mask_dir is not None:
        val_dataset = HistopathDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            target_size=target_size
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader
