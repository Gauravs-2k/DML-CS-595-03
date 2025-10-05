import torch
import torch.nn as nn
from torchvision import models
import os

class ModelUtils:

    @staticmethod
    def setup_model(model_name: str, num_classes: int, device: torch.device):
        # printing setup message
        print(f"Setting up {model_name} model...")

        model = None

        # creating model based on name
        if model_name == "alexnet":
            model = models.alexnet(weights=None)
            model.features[0] = nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=3, stride=1, padding=1
            )
            model.classifier[-1] = nn.Linear(
                in_features=4096, out_features=num_classes
            )
        elif model_name == "resnet18":
            model = models.resnet18(weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()

        if model is None:
            raise ValueError(f"Unsupported model: {model_name}")

        # moving model to device
        model = model.to(device)
        print(f"Model setup complete. Using {model_name} with {num_classes} classes")

        return model

    @staticmethod
    def save_model(model: nn.Module, model_name: str, num_classes: int,
                   dataset_name: str, training_history: dict, save_path: str = None):
        # setting save path
        if save_path is None:
            os.makedirs('models', exist_ok=True)
            save_path = f'models/federated_model_{model_name}_{dataset_name}.pth'

        # saving model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'num_classes': num_classes,
            'dataset_name': dataset_name,
            'training_history': training_history
        }, save_path)

        print(f"Model saved to {save_path}")
        return save_path

    @staticmethod
    def load_model(model_path: str, device: torch.device):
        # checking if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # loading checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # extracting info
        model_name = checkpoint['model_name']
        num_classes = checkpoint['num_classes']
        dataset_name = checkpoint['dataset_name']

        # setting up model
        model = ModelUtils.setup_model(model_name, num_classes, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        training_history = checkpoint.get('training_history', {})

        print(f"Model loaded from {model_path}")
        print(f"Model: {model_name}, Classes: {num_classes}, Dataset: {dataset_name}")

        return model, model_name, num_classes, dataset_name, training_history

    @staticmethod
    def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                      device: torch.device, criterion=None):
        # setting criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # setting model to eval
        model.eval()

        # initializing metrics
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # calculating accuracy and loss
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss

    @staticmethod
    def get_model_info(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }