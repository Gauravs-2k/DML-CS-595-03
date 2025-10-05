from typing import Dict, Any
class Config:
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
        # setting default config
        self.config = self.DEFAULT_CONFIG.copy()
        # updating with provided kwargs
        self.config.update(kwargs)

        # validating configuration
        self._validate_config()

    def _validate_config(self):
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
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self._validate_config()

    def update(self, config_dict: Dict[str, Any]):
        self.config.update(config_dict)
        self._validate_config()

    def print_config(self):
        print("=== Federated Learning Simulation Configuration ===")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("=" * 50)

default_config = Config()