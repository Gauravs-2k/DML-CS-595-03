"""
Federated Learning Simulation - Main Entry Point

This module serves as the main entry point for running federated learning simulations.
It imports and orchestrates all the modular components.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.federated_learning.config import Config
from src.federated_learning.federated_trainer import FederatedLearningSimulation


def main():
    """Main function to run the federated learning simulation"""
    # Configuration (meets assignment requirements)
    config = Config(
        num_clients=64,              # Assignment prefers 64+ (minimum 16)
        dataset_name='CIFAR10',      # or 'CIFAR100'
        model_name='alexnet',        # or 'resnet18'
        max_workers=8,               # Adjust based on your system
        epochs_per_round=3,          # Reduced for faster training
        num_rounds=15,               # Adjust based on convergence
        learning_rate=0.001
    )

    # Print configuration
    config.print_config()
    # Create and run simulation
    fl_sim = FederatedLearningSimulation(config)
    fl_sim.run_simulation()


if __name__ == "__main__":
    main()