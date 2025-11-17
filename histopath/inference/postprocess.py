"""Post-processing utilities for predictions."""

import numpy as np
from scipy import ndimage
from typing import Optional


class PostProcessor:
    """Post-processing utilities for segmentation masks."""
    
    @staticmethod
    def apply_threshold(
        mask: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Apply threshold to probability mask.
        
        Args:
            mask: Probability mask with values in [0, 1]
            threshold: Threshold value
        
        Returns:
            Binary mask (0 or 1)
        """
        return (mask > threshold).astype(np.uint8)
    
    @staticmethod
    def remove_small_objects(
        mask: np.ndarray,
        min_size: int = 50
    ) -> np.ndarray:
        """
        Remove small connected components.
        
        Args:
            mask: Binary mask
            min_size: Minimum size of objects to keep (in pixels)
        
        Returns:
            Cleaned binary mask
        """
        from skimage.morphology import remove_small_objects as rso
        
        # Convert to boolean
        mask_bool = mask.astype(bool)
        
        # Remove small objects
        cleaned = rso(mask_bool, min_size=min_size)
        
        return cleaned.astype(np.uint8)
    
    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask.
        
        Args:
            mask: Binary mask
        
        Returns:
            Mask with holes filled
        """
        return ndimage.binary_fill_holes(mask).astype(np.uint8)
    
    @staticmethod
    def morphological_closing(
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply morphological closing operation.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
        
        Returns:
            Processed mask
        """
        from skimage.morphology import closing, disk
        
        selem = disk(kernel_size)
        return closing(mask, selem).astype(np.uint8)
    
    @staticmethod
    def morphological_opening(
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply morphological opening operation.
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
        
        Returns:
            Processed mask
        """
        from skimage.morphology import opening, disk
        
        selem = disk(kernel_size)
        return opening(mask, selem).astype(np.uint8)
