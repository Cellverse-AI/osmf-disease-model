"""Data loading utilities for histopathological images."""

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from tqdm import tqdm


class DataLoader:
    """
    Utility class for loading histopathological images from directories.
    
    Supports various image formats (.png, .tif, .jpg) and provides
    batch loading with progress tracking.
    """
    
    @staticmethod
    def load_patches_from_directory(
        directory: str,
        target_size: Tuple[int, int] = (256, 256),
        file_extensions: List[str] = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    ) -> np.ndarray:
        """
        Load all images from a directory.
        
        This function replicates the behavior from the original notebook,
        loading images, resizing them, and returning as a numpy array.
        
        Args:
            directory: Path to directory containing images
            target_size: Target size (height, width) for resizing images
            file_extensions: List of valid file extensions to load
        
        Returns:
            numpy array of shape (N, H, W, C) where:
                N = number of images
                H, W = height and width (target_size)
                C = number of channels (3 for RGB, 1 for grayscale)
        
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid images found in directory
        
        Example:
            >>> loader = DataLoader()
            >>> images = loader.load_patches_from_directory(
            ...     "data/train/images",
            ...     target_size=(256, 256)
            ... )
            >>> print(images.shape)  # (N, 256, 256, 3)
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get list of valid image files
        filenames = sorted([
            f for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in file_extensions)
        ])
        
        if not filenames:
            raise ValueError(f"No valid images found in {directory}")
        
        patches = []
        
        # Load images with progress bar
        for filename in tqdm(filenames, desc=f"Loading from {os.path.basename(directory)}"):
            image_path = os.path.join(directory, filename)
            try:
                img = DataLoader.load_image(image_path, target_size)
                patches.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
        
        if not patches:
            raise ValueError(f"Failed to load any images from {directory}")
        
        return np.array(patches)
    
    @staticmethod
    def load_image(
        path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Load a single image from file.
        
        Args:
            path: Path to image file
            target_size: Optional target size (height, width) for resizing
        
        Returns:
            numpy array of shape (H, W, C) or (H, W) for grayscale
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        
        Example:
            >>> img = DataLoader.load_image("image.png", target_size=(256, 256))
            >>> print(img.shape)  # (256, 256, 3)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            # Open image with PIL
            img = Image.open(path)
            
            # Resize if target size specified
            if target_size is not None:
                # PIL uses (width, height), we use (height, width)
                img = img.resize((target_size[1], target_size[0]))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Failed to load image {path}: {e}")
    
    @staticmethod
    def get_image_filenames(
        directory: str,
        file_extensions: List[str] = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    ) -> List[str]:
        """
        Get list of image filenames in directory.
        
        Args:
            directory: Path to directory
            file_extensions: List of valid file extensions
        
        Returns:
            Sorted list of image filenames
        
        Example:
            >>> filenames = DataLoader.get_image_filenames("data/images/")
            >>> print(len(filenames))
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        filenames = sorted([
            f for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in file_extensions)
        ])
        
        return filenames
    
    @staticmethod
    def load_image_mask_pair(
        image_path: str,
        mask_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an image and its corresponding mask.
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            target_size: Optional target size for resizing
        
        Returns:
            Tuple of (image, mask) as numpy arrays
        
        Example:
            >>> img, mask = DataLoader.load_image_mask_pair(
            ...     "data/images/img001.png",
            ...     "data/masks/img001.png",
            ...     target_size=(256, 256)
            ... )
        """
        image = DataLoader.load_image(image_path, target_size)
        mask = DataLoader.load_image(mask_path, target_size)
        
        return image, mask
    
    @staticmethod
    def verify_image_mask_correspondence(
        image_dir: str,
        mask_dir: str
    ) -> Tuple[List[str], List[str]]:
        """
        Verify that images and masks have corresponding files.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
        
        Returns:
            Tuple of (matched_image_files, matched_mask_files)
        
        Raises:
            ValueError: If no matching pairs found
        """
        image_files = DataLoader.get_image_filenames(image_dir)
        mask_files = DataLoader.get_image_filenames(mask_dir)
        
        # Find matching filenames (ignoring extensions)
        image_names = {os.path.splitext(f)[0]: f for f in image_files}
        mask_names = {os.path.splitext(f)[0]: f for f in mask_files}
        
        common_names = set(image_names.keys()) & set(mask_names.keys())
        
        if not common_names:
            raise ValueError(
                f"No matching image-mask pairs found between "
                f"{image_dir} and {mask_dir}"
            )
        
        matched_images = [image_names[name] for name in sorted(common_names)]
        matched_masks = [mask_names[name] for name in sorted(common_names)]
        
        print(f"Found {len(matched_images)} matching image-mask pairs")
        
        return matched_images, matched_masks
