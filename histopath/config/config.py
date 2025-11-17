"""Configuration management for histopathology analysis."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for histopathology analysis.
    
    Loads configuration from YAML files and provides access to parameters.
    Supports default values and parameter validation.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        config_path (Optional[str]): Path to configuration file
    
    Example:
        >>> config = Config("configs/segmentation_config.yaml")
        >>> batch_size = config.get("data.batch_size", default=16)
        >>> config.update({"training.epochs": 100})
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        elif config_path:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            "data": {
                "image_size": [256, 256],
                "batch_size": 16,
                "num_workers": 4,
                "train_split": 0.8,
                "normalize": True,
            },
            "model": {
                "architecture": "cbam_unet",
                "in_channels": 3,
                "out_channels": 1,
                "base_channels": 32,
                "depth": 4,
            },
            "training": {
                "epochs": 50,
                "learning_rate": 0.0001,
                "optimizer": "adam",
                "loss_weights": {
                    "bce": 0.0,
                    "dice": 0.3,
                    "jaccard": 0.7,
                },
                "device": "cuda",  # or "cpu"
            },
            "paths": {
                "data_dir": "data/",
                "checkpoint_dir": "checkpoints/",
                "log_dir": "logs/",
                "output_dir": "outputs/",
            },
        }
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file and merge with defaults.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._deep_update(self.config, file_config)
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports nested keys using dot notation (e.g., "data.batch_size").
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get("data.batch_size", default=16)
            16
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        
        Example:
            >>> config.update({"training.epochs": 100})
        """
        self._deep_update(self.config, updates)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate data parameters
            assert self.get("data.batch_size") > 0, "Batch size must be positive"
            assert 0 < self.get("data.train_split") <= 1, "Train split must be in (0, 1]"
            
            # Validate model parameters
            assert self.get("model.in_channels") > 0, "Input channels must be positive"
            assert self.get("model.out_channels") > 0, "Output channels must be positive"
            
            # Validate training parameters
            assert self.get("training.epochs") > 0, "Epochs must be positive"
            assert self.get("training.learning_rate") > 0, "Learning rate must be positive"
            
            # Validate loss weights sum to reasonable value
            loss_weights = self.get("training.loss_weights")
            total_weight = sum(loss_weights.values())
            assert total_weight > 0, "Loss weights must sum to positive value"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Configuration saved to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path={self.config_path})"
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        return yaml.dump(self.config, default_flow_style=False)
