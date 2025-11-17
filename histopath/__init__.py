"""
Histopathology Image Analysis Package

A Python package for histopathological image analysis including:
- Cell density segmentation using CBAM U-Net
- Fibrosis classification with H&E stain separation
- Training and inference pipelines
"""

__version__ = "0.1.0"
__author__ = "Histopathology Analysis Team"

from histopath.config import Config

__all__ = ["Config"]
