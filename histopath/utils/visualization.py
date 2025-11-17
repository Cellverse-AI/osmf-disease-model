"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


class Visualizer:
    """Visualization utilities for segmentation."""
    
    @staticmethod
    def plot_segmentation(
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot image, ground truth mask, and prediction.
        
        Args:
            image: Input image (H, W, C)
            mask: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            save_path: Path to save figure
        """
        n_plots = 1 + (mask is not None) + (prediction is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        
        # Plot image
        axes[idx].imshow(image)
        axes[idx].set_title('Image')
        axes[idx].axis('off')
        idx += 1
        
        # Plot ground truth
        if mask is not None:
            axes[idx].imshow(mask, cmap='gray')
            axes[idx].set_title('Ground Truth')
            axes[idx].axis('off')
            idx += 1
        
        # Plot prediction
        if prediction is not None:
            axes[idx].imshow(prediction, cmap='gray')
            axes[idx].set_title('Prediction')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history curves.
        
        Args:
            history: Dictionary with training history
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot metrics
        if 'train_dice' in history:
            axes[1].plot(history['train_dice'], label='Train Dice')
        if 'val_dice' in history:
            axes[1].plot(history['val_dice'], label='Val Dice')
        if 'train_iou' in history:
            axes[1].plot(history['train_iou'], label='Train IoU', linestyle='--')
        if 'val_iou' in history:
            axes[1].plot(history['val_iou'], label='Val IoU', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Training and Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_attention_maps(
        image: np.ndarray,
        attention_maps: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention maps.
        
        Args:
            image: Input image
            attention_maps: List of attention maps
            save_path: Path to save figure
        """
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(1, n_maps + 1, figsize=(5 * (n_maps + 1), 5))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot attention maps
        for idx, att_map in enumerate(attention_maps):
            axes[idx + 1].imshow(att_map, cmap='hot')
            axes[idx + 1].set_title(f'Attention Map {idx + 1}')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
