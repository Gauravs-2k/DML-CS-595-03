import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


class TrainingLogger:

    def __init__(self):
        # initializing logs and history
        self.training_logs = []
        self.training_history = {
            'round': [],
            'global_accuracy': [],
            'global_loss': []
        }

    def log_batch(self, timestamp: str, round_num: int, batch_num: int,
                  client_id: int, epoch: int, train_loss: float,
                  train_acc: float, batch_size: int):
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
        self.training_history['round'].append(round_num)
        self.training_history['global_accuracy'].append(global_accuracy)
        self.training_history['global_loss'].append(global_loss)

    def save_logs(self, output_dir: str = 'data'):
        # creating output directory
        os.makedirs(output_dir, exist_ok=True)

        # creating dataframe
        df = pd.DataFrame(self.training_logs)
        # saving to csv
        df.to_csv(f'{output_dir}/training_results.csv', index=False)

        # saving to json
        df.to_json(f'{output_dir}/training_results.json', orient='records')
        # saving to parquet
        df.to_parquet(f'{output_dir}/training_results.parquet')

        # creating summary
        summary_df = pd.DataFrame(self.training_history)
        summary_df.to_csv(f'{output_dir}/training_summary.csv', index=False)

        print(f"Training data saved to '{output_dir}/' directory")
        print(f"Total training records: {len(self.training_logs)}")

        return df, summary_df

    def get_logs_dataframe(self):
        return pd.DataFrame(self.training_logs)

    def get_summary_dataframe(self):
        return pd.DataFrame(self.training_history)


class PlotUtils:

    @staticmethod
    def plot_training_history(training_history: dict, save_path: str = 'static/training_history.png'):
        # creating subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # plotting accuracy
        ax1.plot(training_history['round'], training_history['global_accuracy'])
        ax1.set_title('Global Model Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True)

        # plotting loss
        ax2.plot(training_history['round'], training_history['global_loss'])
        ax2.set_title('Global Model Loss')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        plt.tight_layout()

        # creating directory and saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training history plot saved to '{save_path}'")

    @staticmethod
    def plot_client_performance(training_logs_df: pd.DataFrame,
                               save_path: str = 'static/client_performance.png'):
        # checking if data exists
        if training_logs_df.empty:
            return

        # aggregating client stats
        client_stats = training_logs_df.groupby('client_id').agg({
            'train_loss': ['mean', 'std'],
            'train_acc': ['mean', 'std']
        }).round(4)

        # creating subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # plotting loss
        client_stats['train_loss']['mean'].plot(kind='bar', ax=ax1, yerr=client_stats['train_loss']['std'])
        ax1.set_title('Average Training Loss by Client')
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='x', rotation=45)

        # plotting accuracy
        client_stats['train_acc']['mean'].plot(kind='bar', ax=ax2, yerr=client_stats['train_acc']['std'])
        ax2.set_title('Average Training Accuracy by Client')
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # creating directory and saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Client performance plot saved to '{save_path}'")


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_directories():
    directories = ['data', 'models', 'static']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)


def print_config(config: dict):
    print("=== Federated Learning Simulation Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)