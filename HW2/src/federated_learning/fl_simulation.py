import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.federated_learning.config import Config
from src.federated_learning.federated_trainer import FederatedLearningSimulation


def main():
    """Main function to run the federated learning simulation"""
    # configuring simulation parameters
    config = Config(
        num_clients=16,             
        dataset_name='CIFAR10',     
        model_name='alexnet',        
        max_workers=8,              
        epochs_per_round=3,         
        num_rounds=15,              
        learning_rate=0.01
    )
    # displaying configuration
    config.print_config()
    # running federated learning simulation
    fl_sim = FederatedLearningSimulation(config)
    fl_sim.run_simulation()


if __name__ == "__main__":
    main()