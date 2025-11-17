"""Inference engine for trained segmentation models."""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm

from histopath.data.loader import DataLoader
from histopath.data.transforms import Transforms
from histopath.config import Config


class Predictor:
    """
    Inference engine for trained segmentation models.
    
    Handles prediction on single images, batches, or entire directories.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            config: Configuration object
            device: Device to run inference on
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
        print(f"Predictor initialized on device: {device}")
    
    def predict_single(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict on single image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            threshold: Threshold for binary segmentation
        
        Returns:
            Binary mask as numpy array (H, W)
        """
        # Normalize and convert to tensor
        if image.max() > 1.0:
            image = Transforms.normalize(image)
        
        image_tensor = Transforms.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
        
        # Convert to numpy and apply threshold
        mask = output.squeeze().cpu().numpy()
        mask = (mask > threshold).astype(np.uint8)
        
        return mask
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Predict on batch of images.
        
        Args:
            images: List of images as numpy arrays
            threshold: Threshold for binary segmentation
        
        Returns:
            List of binary masks
        """
        masks = []
        
        for image in tqdm(images, desc="Predicting"):
            mask = self.predict_single(image, threshold)
            masks.append(mask)
        
        return masks
    
    def predict_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        threshold: float = 0.5,
        save_format: str = 'png'
    ) -> None:
        """
        Predict on all images in directory and save results.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save predictions
            threshold: Threshold for binary segmentation
            save_format: Output format ('png' or 'npy')
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_files = DataLoader.get_image_filenames(input_dir)
        
        print(f"Found {len(image_files)} images in {input_dir}")
        print(f"Saving predictions to {output_dir}")
        
        # Process each image
        for filename in tqdm(image_files, desc="Processing images"):
            # Load image
            image_path = os.path.join(input_dir, filename)
            image = DataLoader.load_image(
                image_path,
                target_size=tuple(self.config.get('data.image_size', [256, 256]))
            )
            
            # Predict
            mask = self.predict_single(image, threshold)
            
            # Save prediction
            base_name = os.path.splitext(filename)[0]
            
            if save_format == 'png':
                from PIL import Image
                output_path = os.path.join(output_dir, f"{base_name}_mask.png")
                mask_img = Image.fromarray(mask * 255)
                mask_img.save(output_path)
            elif save_format == 'npy':
                output_path = os.path.join(output_dir, f"{base_name}_mask.npy")
                np.save(output_path, mask)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        
        print(f"Predictions saved to {output_dir}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: type,
        config: Config,
        device: Optional[torch.device] = None
    ) -> 'Predictor':
        """
        Create predictor from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_class: Model class to instantiate
            config: Configuration object
            device: Device to run on
        
        Returns:
            Predictor instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = model_class(
            in_channels=config.get('model.in_channels', 3),
            out_channels=config.get('model.out_channels', 1),
            base_channels=config.get('model.base_channels', 32)
        )
        
        # Load weights
        model.load_weights(checkpoint_path, device)
        
        return cls(model, config, device)
