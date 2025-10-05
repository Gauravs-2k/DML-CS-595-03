"""
Configuration Module for Federated Learning

This module contains configuration settings and parameter management.
"""

from typing import Dict, Any


class Config:
    """Configuration class for federated learning simulation"""

    DEFAULT_CONFIG = {
        'num_clients': 64,              
        'dataset_name': 'CIFAR10',      
        'model_name': 'alexnet',       
        'max_workers': 8,             
        'epochs_per_round': 3,        
        'num_rounds': 15,              
        'learning_rate': 0.01,        
        'device': 'auto',               
        'batch_size': 32,              
        'random_seed': 42,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,
        'lr_scheduler_step': 3,
        'lr_scheduler_gamma': 0.5     
    }

    def __init__(self, **kwargs):
        """Initialize configuration with defaults and overrides"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config['num_clients'] < 16:
            raise ValueError("Number of clients must be at least 16")

        if self.config['dataset_name'] not in ['CIFAR10', 'CIFAR100']:
            raise ValueError("Dataset must be 'CIFAR10' or 'CIFAR100'")

        if self.config['model_name'] not in ['alexnet', 'resnet18']:
            raise ValueError("Model must be 'alexnet' or 'resnet18'")

        if self.config['max_workers'] < 1:
            raise ValueError("max_workers must be at least 1")

        if self.config['epochs_per_round'] < 1:
            raise ValueError("epochs_per_round must be at least 1")

        if self.config['num_rounds'] < 1:
            raise ValueError("num_rounds must be at least 1")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self._validate_config()

    def update(self, config_dict: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(config_dict)
        self._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()

    def print_config(self):
        """Print configuration in formatted way"""
        print("=== Federated Learning Simulation Configuration ===")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("=" * 50)

    @classmethod
    def get_fast_config(cls) -> 'Config':
        """Get configuration optimized for fast testing"""
        return cls(
            num_clients=16,
            epochs_per_round=1,
            num_rounds=5,
            max_workers=4
        )

    @classmethod
    def get_full_config(cls) -> 'Config':
        """Get full configuration for production runs"""
        return cls(
            num_clients=64,
            epochs_per_round=5,
            num_rounds=20,
            max_workers=8
        )

    @classmethod
    def get_cifar100_config(cls) -> 'Config':
        """Get configuration for CIFAR-100 dataset"""
        return cls(
            dataset_name='CIFAR100',
            num_clients=64,
            epochs_per_round=3,
            num_rounds=15
        )

    @classmethod
    def get_resnet_config(cls) -> 'Config':
        """Get configuration using ResNet18 model"""
        return cls(
            model_name='resnet18',
            num_clients=64,
            epochs_per_round=3,
            num_rounds=15
        )


default_config = Config()