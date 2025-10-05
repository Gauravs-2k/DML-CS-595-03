"""
Data Handler Module for Federated Learning

This module handles dataset loading, partitioning, and visualization for federated learning.
"""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


class DataHandler:
    """Handles dataset loading, partitioning, and visualization for federated learning"""

    def __init__(self, dataset_name: str = "CIFAR10", num_clients: int = 64):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_classes = 10 if dataset_name == "CIFAR10" else 100

        # Initialize datasets
        self.train_dataset = None
        self.test_dataset = None
        self.test_loader = None
        self.client_data = {}

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup_data(self):
        """Setup and partition the dataset for federated learning"""
        print("Setting up data...")

        # Load dataset
        if self.dataset_name == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True, transform=self.train_transform
            )
            self.test_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=self.test_transform
            )
        elif self.dataset_name == "CIFAR100":
            self.train_dataset = datasets.CIFAR100(
                root='./data', train=True, download=True, transform=self.train_transform
            )
            self.test_dataset = datasets.CIFAR100(
                root='./data', train=False, download=True, transform=self.test_transform
            )

        # Create test loader
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

        # Partition data among clients (non-IID)
        self._create_non_iid_partitions()

        print(f"Data setup complete. {len(self.train_dataset)} training samples, "
              f"{len(self.test_dataset)} test samples, {self.num_clients} clients")

        return self.train_dataset, self.test_dataset, self.test_loader, self.client_data

    def _create_non_iid_partitions(self):
        """Create non-IID data partitions for clients"""
        print("Creating non-IID data partitions...")

        # Get labels for all training data
        labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])

        # Group indices by class
        class_indices = {i: np.where(labels == i)[0] for i in range(self.num_classes)}

        # Create non-IID distribution using Dirichlet distribution
        client_data_indices = defaultdict(list)

        for class_id in range(self.num_classes):
            proportions = np.random.dirichlet(np.repeat(0.5, self.num_clients))
            class_data = class_indices[class_id]
            np.random.shuffle(class_data)

            class_counts = np.random.multinomial(len(class_data), proportions)

            start_idx = 0
            for client_id, count in enumerate(class_counts):
                end_idx = start_idx + count
                if count > 0:
                    client_data_indices[client_id].extend(class_data[start_idx:end_idx])
                start_idx = end_idx

        empty_clients = [cid for cid, indices in client_data_indices.items() if len(indices) == 0]
        if empty_clients:
            donor_clients = sorted(
                client_data_indices.items(),
                key=lambda item: len(item[1]),
                reverse=True
            )

            for empty_client in empty_clients:
                for donor_client, donor_indices in donor_clients:
                    if len(donor_indices) > 1:
                        client_data_indices[empty_client].append(donor_indices.pop())
                        break

        self.client_data = {}
        for client_id in range(self.num_clients):
            self.client_data[client_id] = Subset(
                self.train_dataset,
                client_data_indices[client_id]
            )

        self._visualize_data_distribution(client_data_indices)

    def _visualize_data_distribution(self, client_data_indices):
        """Visualize the data distribution across clients"""
        distribution_matrix = np.zeros((self.num_clients, self.num_classes))

        for client_id in range(self.num_clients):
            indices = client_data_indices[client_id]
            for idx in indices:
                label = self.train_dataset[idx][1]
                distribution_matrix[client_id, label] += 1

        plt.figure(figsize=(12, 8))
        sns.heatmap(distribution_matrix,
                   xticklabels=range(self.num_classes),
                   yticklabels=range(self.num_clients),
                   cmap='Blues',
                   annot=False)
        plt.title(f'Data Distribution Across {self.num_clients} Clients')
        plt.xlabel('Class Labels')
        plt.ylabel('Client ID')
        plt.tight_layout()

        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Data distribution visualization saved to 'static/data_distribution.png'")

    def get_client_data_info(self):
        """Get information about client data distribution"""
        client_info = {}
        for client_id, subset in self.client_data.items():
            client_info[client_id] = {
                'num_samples': len(subset),
                'class_distribution': self._get_class_distribution(subset)
            }
        return client_info

    def _get_class_distribution(self, subset):
        """Get class distribution for a subset"""
        labels = [self.train_dataset[idx][1] for idx in subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))