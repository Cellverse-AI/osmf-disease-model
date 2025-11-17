"""File I/O utilities."""

import os
import numpy as np
from PIL import Image
from typing import Optional


class IOUtils:
    """File I/O utilities."""
    
    @staticmethod
    def save_mask(
        mask: np.ndarray,
        path: str,
        format: str = 'png'
    ) -> None:
        """
        Save mask to file.
        
        Args:
            mask: Mask array
            path: Output path
            format: Output format ('png' or 'npy')
        """
        IOUtils.ensure_dir(os.path.dirname(path))
        
        if format == 'png':
            # Convert to uint8 if needed
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            img = Image.fromarray(mask)
            img.save(path)
        elif format == 'npy':
            np.save(path, mask)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_mask(path: str) -> np.ndarray:
        """
        Load mask from file.
        
        Args:
            path: Path to mask file
        
        Returns:
            Mask array
        """
        if path.endswith('.npy'):
            return np.load(path)
        else:
            img = Image.open(path)
            return np.array(img)
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            path: Directory path
        """
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
