"""Training orchestrator for segmentation models."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm

from histopath.training.metrics import MetricsTracker
from histopath.config import Config


class Trainer:
    """
    Training orchestrator for segmentation models.
    
    Handles the complete training loop including:
    - Training epochs with progress tracking
    - Validation
    - Checkpoint saving and loading
    - Metric logging
    - Early stopping
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration object
        device: Device to train on (cuda/cpu)
        scheduler: Optional learning rate scheduler
    
    Attributes:
        model: The model being trained
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration
        device: Training device
        scheduler: LR scheduler
        history: Training history
        best_val_loss: Best validation loss seen
        epochs_without_improvement: Counter for early stopping
    
    Example:
        >>> from histopath.models.segmentation import CBAMUNet
        >>> from histopath.training.losses import BCEDiceJaccardLoss
        >>> 
        >>> model = CBAMUNet()
        >>> criterion = BCEDiceJaccardLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     config=config,
        ...     device=device
        ... )
        >>> 
        >>> history = trainer.train(num_epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Config,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            config: Configuration object
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }
        
        # For early stopping and checkpointing
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create checkpoint directory
        checkpoint_dir = config.get('paths.checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            metrics_tracker.update(outputs.detach(), masks.detach())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics_tracker.get_current()["dice"]:.4f}'
            })
        
        # Compute average metrics
        avg_loss = running_loss / len(self.train_loader)
        avg_metrics = metrics_tracker.get_average()
        
        return {
            'loss': avg_loss,
            'dice': avg_metrics['dice'],
            'iou': avg_metrics['iou']
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        metrics_tracker = MetricsTracker()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc='Validation')
        
        with torch.no_grad():
            for images, masks in pbar:
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Track metrics
                running_loss += loss.item()
                metrics_tracker.update(outputs, masks)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics_tracker.get_current()["dice"]:.4f}'
                })
        
        # Compute average metrics
        avg_loss = running_loss / len(self.val_loader)
        avg_metrics = metrics_tracker.get_average()
        
        return {
            'loss': avg_loss,
            'dice': avg_metrics['dice'],
            'iou': avg_metrics['iou']
        }
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            Training history dictionary
        
        Example:
            >>> history = trainer.train(num_epochs=50)
            >>> print(f"Best validation loss: {min(history['val_loss']):.4f}")
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Dice: {train_metrics['dice']:.4f} | "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                
                # Log validation metrics
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_dice'].append(val_metrics['dice'])
                self.history['val_iou'].append(val_metrics['iou'])
                
                print(f"Val Loss:   {val_metrics['loss']:.4f} | "
                      f"Dice: {val_metrics['dice']:.4f} | "
                      f"IoU: {val_metrics['iou']:.4f}")
                
                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    best_path = os.path.join(
                        self.config.get('paths.checkpoint_dir', 'checkpoints'),
                        'best_model.pth'
                    )
                    self.save_checkpoint(epoch + 1, val_metrics, best_path)
                    print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                patience = self.config.get('training.early_stopping_patience', 10)
                if self.epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
            
            # Save periodic checkpoint
            save_freq = self.config.get('training.save_frequency', 5)
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.get('paths.checkpoint_dir', 'checkpoints'),
                    f'checkpoint_epoch_{epoch + 1}.pth'
                )
                metrics = val_metrics if self.val_loader else train_metrics
                self.save_checkpoint(epoch + 1, metrics, checkpoint_path)
        
        print("\n" + "=" * 70)
        print("Training completed!")
        if self.val_loader:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        path: str
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and resume training.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Epoch number to resume from
        
        Example:
            >>> trainer = Trainer(...)
            >>> start_epoch = trainer.load_checkpoint('checkpoints/checkpoint.pth')
            >>> trainer.train(num_epochs=50)  # Continues from start_epoch
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'metrics' in checkpoint:
            val_loss = checkpoint['metrics'].get('loss', float('inf'))
            self.best_val_loss = min(self.best_val_loss, val_loss)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return self.history
