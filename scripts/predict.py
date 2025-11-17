"""Inference script for segmentation model."""

import argparse
import torch

from histopath.config import Config
from histopath.models.segmentation import CBAMUNet
from histopath.inference.predictor import Predictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained CBAM U-Net model'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save predictions'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary segmentation'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'npy'],
        help='Output format for predictions'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 70)
    print("CBAM U-Net Inference")
    print("=" * 70)
    
    # Load configuration
    config = Config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create predictor from checkpoint
    print(f"\nLoading model from {args.checkpoint}")
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_class=CBAMUNet,
        config=config,
        device=device
    )
    
    # Run prediction
    print(f"\nProcessing images from {args.input_dir}")
    predictor.predict_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        save_format=args.format
    )
    
    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
