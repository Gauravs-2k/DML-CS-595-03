import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time
import copy
from collections import OrderedDict
import random
import torch.backends.cudnn as cudnn
from src.federated_learning.data_handler import DataHandler
from src.federated_learning.model_utils import ModelUtils
from src.federated_learning.utils import TrainingLogger, PlotUtils, get_timestamp, ensure_directories
from src.federated_learning.config import Config


class FederatedLearningSimulation:

    def __init__(self, config: Config = None):
        # setting config
        if config is None:
            config = Config()

        self.config = config
        # initializing logger
        self.logger = TrainingLogger()

        # initializing data and model attributes
        self.data_handler = None
        self.global_model = None
        self.device = None

        self.train_dataset = None
        self.test_dataset = None
        self.test_loader = None
        self.client_data = {}

        # setting random seed
        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            cudnn.deterministic = True
            cudnn.benchmark = False

        # determining device
        device_str = self.config.get('device', 'auto')
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        print(f"Using device: {self.device}")

    def setup_simulation(self):
        """Setup data and model for federated learning"""
        # ensuring directories exist
        ensure_directories()

        # initializing data handler
        self.data_handler = DataHandler(
            dataset_name=self.config.get('dataset_name'),
            num_clients=self.config.get('num_clients')
        )
        # setting up data
        self.train_dataset, self.test_dataset, self.test_loader, self.client_data = self.data_handler.setup_data()

        # setting up global model
        self.global_model = ModelUtils.setup_model(
            model_name=self.config.get('model_name'),
            num_classes=self.data_handler.num_classes,
            device=self.device
        )

    def local_training_job(self, client_id: int, global_weights: dict, round_num: int) -> dict:
        # recording start time
        start_time = time.time()
        timestamp = get_timestamp()

        # creating local model copy
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_weights)
        local_model = local_model.to(self.device)

        # creating data loader for client
        local_loader = DataLoader(
            self.client_data[client_id],
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )

        # setting optimizer
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.get('learning_rate'),
                momentum=self.config.get('momentum', 0.0),
                weight_decay=self.config.get('weight_decay', 0.0),
                nesterov=self.config.get('nesterov', False)
            )
        else:
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=self.config.get('learning_rate'),
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        # setting loss criterion
        criterion = nn.CrossEntropyLoss()

        # setting scheduler if configured
        scheduler = None
        scheduler_step = self.config.get('lr_scheduler_step')
        if scheduler_step and scheduler_step > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_step,
                gamma=self.config.get('lr_scheduler_gamma', 0.1)
            )

        # training loop
        local_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_logs = []

        epochs_per_round = self.config.get('epochs_per_round')

        for epoch in range(epochs_per_round):
            for batch_idx, (data, target) in enumerate(local_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                batch_correct = (predicted == target).sum().item()
                batch_samples = target.size(0)
                batch_accuracy = batch_correct / batch_samples

                total_loss += loss.item()
                total_correct += batch_correct
                total_samples += batch_samples

                # logging batch
                self.logger.log_batch(
                    timestamp=timestamp,
                    round_num=round_num,
                    batch_num=batch_idx,
                    client_id=client_id,
                    epoch=epoch,
                    train_loss=loss.item(),
                    train_acc=batch_accuracy,
                    batch_size=batch_samples
                )

                batch_logs.append({
                    'time': timestamp,
                    'round': round_num,
                    'batch_num': batch_idx,
                    'client_id': client_id,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_acc': batch_accuracy,
                    'batch_size': batch_samples
                })

            # stepping scheduler
            if scheduler is not None:
                scheduler.step()

        # calculating averages
        avg_loss = total_loss / len(batch_logs) if batch_logs else 0.0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # preparing model state
        model_state = {
            key: value.detach().cpu()
            for key, value in local_model.state_dict().items()
        }

        local_model.to('cpu')

        return {
            'client_id': client_id,
            'model_weights': model_state,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'num_samples': len(self.client_data[client_id]),
            'batch_logs': batch_logs,
            'training_time': time.time() - start_time
        }

    def federated_averaging(self, client_results: list) -> dict:
        # initializing aggregated weights
        aggregated_weights = {}
        total_samples = sum(result['num_samples'] for result in client_results)

        with torch.no_grad():
            for result in client_results:
                if result['num_samples'] == 0:
                    continue

                weight = result['num_samples'] / total_samples
                client_weights = result['model_weights']

                for key in client_weights:
                    if key not in aggregated_weights:
                        aggregated_weights[key] = torch.zeros_like(client_weights[key])
                    aggregated_weights[key] += weight * client_weights[key]

        return aggregated_weights

    def run_simulation(self):
        # printing simulation start info
        print(f"Starting federated learning simulation with {self.config.get('num_clients')} clients...")
        print(f"Using ThreadPoolExecutor with max_workers={self.config.get('max_workers')}")

        # setting up simulation
        self.setup_simulation()

        # evaluating initial model
        initial_acc, initial_loss = ModelUtils.evaluate_model(
            self.global_model, self.test_loader, self.device
        )
        print(f"Initial - Accuracy: {initial_acc:.2f}%, Loss: {initial_loss:.4f}")

        # getting config values
        max_workers = self.config.get('max_workers')
        num_rounds = self.config.get('num_rounds')

        # running rounds
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for round_num in range(1, num_rounds + 1):
                print(f"\n--- Round {round_num}/{num_rounds} ---")

                # getting global weights
                global_weights = OrderedDict(
                    (key, value.detach().cpu())
                    for key, value in self.global_model.state_dict().items()
                )

                # submitting training jobs
                futures = []
                num_clients = self.config.get('num_clients')
                for client_id in range(num_clients):
                    future = executor.submit(
                        self.local_training_job,
                        client_id,
                        global_weights,
                        round_num
                    )
                    futures.append(future)

                # collecting results
                client_results = []
                for future in as_completed(futures):
                    result = future.result()
                    client_results.append(result)

                # performing federated averaging
                aggregated_weights = self.federated_averaging(client_results)
                self.global_model.load_state_dict(aggregated_weights)

                # evaluating global model
                accuracy, loss = ModelUtils.evaluate_model(
                    self.global_model, self.test_loader, self.device
                )
                print(f"Global Model - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")

                # logging round
                self.logger.log_round(round_num, accuracy, loss)

                # calculating client averages
                avg_client_loss = np.mean([r['loss'] for r in client_results])
                avg_client_accuracy = np.mean([r['accuracy'] for r in client_results])
                print(f"Average Client Loss: {avg_client_loss:.4f}, Accuracy: {avg_client_accuracy:.2f}%")

        # saving results
        self.save_results()

        print(f"\nSimulation complete! Final accuracy: {accuracy:.2f}%")

    def save_results(self):
        # saving logs
        self.logger.save_logs()

        # saving model
        ModelUtils.save_model(
            model=self.global_model,
            model_name=self.config.get('model_name'),
            num_classes=self.data_handler.num_classes,
            dataset_name=self.config.get('dataset_name'),
            training_history=self.logger.training_history
        )

        # plotting training history
        PlotUtils.plot_training_history(self.logger.training_history)

        # plotting client performance
        training_df = self.logger.get_logs_dataframe()
        PlotUtils.plot_client_performance(training_df)