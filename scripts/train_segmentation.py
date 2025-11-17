"""Training script for CBAM U-Net segmentation model."""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from histopath.config import Config
from histopath.models.segmentation import CBAMUNet
from histopath.data.dataset import HistopathDataset
from histopath.training.losses import BCEDiceJaccardLoss
from histopath.training.trainer import Trainer
from histopath.utils.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CBAM U-Net for histopathological image segmentation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--train-images',
        type=str,
        required=True,
        help='Path to training images directory'
    )
    
    parser.add_argument(
        '--train-masks',
        type=str,
        required=True,
        help='Path to training masks directory'
    )
    
    parser.add_argument(
        '--val-images',
        type=str,
        default=None,
        help='Path to validation images directory'
    )
    
    parser.add_argument(
        '--val-masks',
        type=str,
        default=None,
        help='Path to validation masks directory'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and logs'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("=" * 70)
    print("CBAM U-Net Training")
    print("=" * 70)
    
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.update({'training.epochs': args.epochs})
    if args.batch_size is not None:
        config.update({'data.batch_size': args.batch_size})
    if args.lr is not None:
        config.update({'training.learning_rate': args.lr})
    if args.output_dir:
        config.update({
            'paths.checkpoint_dir': os.path.join(args.output_dir, 'checkpoints'),
            'paths.log_dir': os.path.join(args.output_dir, 'logs')
        })
    
    # Validate configuration
    if not config.validate():
        print("Error: Invalid configuration")
        return
    
    print("\nConfiguration:")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Batch size: {config.get('data.batch_size')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Image size: {config.get('data.image_size')}")
    print(f"  Device: {config.get('training.device')}")
    
    # Set device
    device_name = config.get('training.device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, using CPU")
    
    print(f"\nUsing device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = HistopathDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        target_size=tuple(config.get('data.image_size')),
        normalize=config.get('data.normalize', True)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('data.batch_size'),
        shuffle=True,
        num_workers=config.get('data.num_workers', 4),
        pin_memory=True
    )
    
    # Create validation dataset if provided
    val_loader = None
    if args.val_images and args.val_masks:
        val_dataset = HistopathDataset(
            image_dir=args.val_images,
            mask_dir=args.val_masks,
            target_size=tuple(config.get('data.image_size')),
            normalize=config.get('data.normalize', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('data.batch_size'),
            shuffle=False,
            num_workers=config.get('data.num_workers', 4),
            pin_memory=True
        )
    
    # Create model
    print("\nInitializing model...")
    model = CBAMUNet(
        in_channels=config.get('model.in_channels'),
        out_channels=config.get('model.out_channels'),
        base_channels=config.get('model.base_channels')
    )
    
    model.summary()
    
    # Create loss function
    loss_weights = config.get('training.loss_weights')
    criterion = BCEDiceJaccardLoss(
        bce_weight=loss_weights.get('bce', 0.0),
        dice_weight=loss_weights.get('dice', 0.3),
        jaccard_weight=loss_weights.get('jaccard', 0.7)
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('training.learning_rate')
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        start_epoch = trainer.load_checkpoint(args.checkpoint)
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    history = trainer.train(num_epochs=config.get('training.epochs'))
    
    # Save final model
    final_path = os.path.join(
        config.get('paths.checkpoint_dir'),
        'final_model.pth'
    )
    model.save_weights(final_path, epoch=config.get('training.epochs'))
    print(f"\nSaved final model to {final_path}")
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_path = os.path.join(
        config.get('paths.output_dir', 'outputs'),
        'training_history.png'
    )
    
    try:
        Visualizer.plot_training_history(history, save_path=plot_path)
        print(f"Saved training history plot to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save training plot: {e}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
