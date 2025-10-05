"""
Utilities Module for Federated Learning

This module contains utility functions for logging, plotting, and helper operations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


class TrainingLogger:
    """Handles detailed training logging for federated learning"""

    def __init__(self):
        self.training_logs = []
        self.training_history = {
            'round': [],
            'global_accuracy': [],
            'global_loss': []
        }

    def log_batch(self, timestamp: str, round_num: int, batch_num: int,
                  client_id: int, epoch: int, train_loss: float,
                  train_acc: float, batch_size: int):
        """Log a single training batch"""
        self.training_logs.append({
            'time': timestamp,
            'round': round_num,
            'batch_num': batch_num,
            'client_id': client_id,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'batch_size': batch_size
        })

    def log_round(self, round_num: int, global_accuracy: float, global_loss: float):
        """Log global model performance for a round"""
        self.training_history['round'].append(round_num)
        self.training_history['global_accuracy'].append(global_accuracy)
        self.training_history['global_loss'].append(global_loss)

    def save_logs(self, output_dir: str = 'data'):
        """Save training logs to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Convert training logs to DataFrame and save as CSV
        df = pd.DataFrame(self.training_logs)
        df.to_csv(f'{output_dir}/training_results.csv', index=False)

        # Also save as other formats for convenience
        df.to_json(f'{output_dir}/training_results.json', orient='records')
        df.to_parquet(f'{output_dir}/training_results.parquet')

        # Save summary statistics
        summary_df = pd.DataFrame(self.training_history)
        summary_df.to_csv(f'{output_dir}/training_summary.csv', index=False)

        print(f"Training data saved to '{output_dir}/' directory")
        print(f"Total training records: {len(self.training_logs)}")

        return df, summary_df

    def get_logs_dataframe(self):
        """Get training logs as pandas DataFrame"""
        return pd.DataFrame(self.training_logs)

    def get_summary_dataframe(self):
        """Get training summary as pandas DataFrame"""
        return pd.DataFrame(self.training_history)


class PlotUtils:
    """Utilities for plotting training results and visualizations"""

    @staticmethod
    def plot_training_history(training_history: dict, save_path: str = 'static/training_history.png'):
        """Plot training accuracy and loss over rounds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        ax1.plot(training_history['round'], training_history['global_accuracy'])
        ax1.set_title('Global Model Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True)

        # Loss plot
        ax2.plot(training_history['round'], training_history['global_loss'])
        ax2.set_title('Global Model Loss')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        plt.tight_layout()

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training history plot saved to '{save_path}'")

    @staticmethod
    def plot_client_performance(training_logs_df: pd.DataFrame,
                               save_path: str = 'static/client_performance.png'):
        """Plot performance metrics by client"""
        if training_logs_df.empty:
            return

        # Group by client and calculate averages
        client_stats = training_logs_df.groupby('client_id').agg({
            'train_loss': ['mean', 'std'],
            'train_acc': ['mean', 'std']
        }).round(4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss by client
        client_stats['train_loss']['mean'].plot(kind='bar', ax=ax1, yerr=client_stats['train_loss']['std'])
        ax1.set_title('Average Training Loss by Client')
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='x', rotation=45)

        # Accuracy by client
        client_stats['train_acc']['mean'].plot(kind='bar', ax=ax2, yerr=client_stats['train_acc']['std'])
        ax2.set_title('Average Training Accuracy by Client')
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Client performance plot saved to '{save_path}'")


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = ['data', 'models', 'static']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)


def print_config(config: dict):
    """Print configuration in a formatted way"""
    print("=== Federated Learning Simulation Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)