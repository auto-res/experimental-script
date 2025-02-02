import yaml
import torch
import torch.nn as nn
from pathlib import Path
from .preprocess import get_mnist_loaders
from .train import train_epoch
from .evaluate import evaluate
from .optimizers.amadgrad import AMADGRADOptimizer

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Load config
    config_path = Path('config/default.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup data
    train_loader, test_loader = get_mnist_loaders(
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )
    
    # Setup model
    model = SimpleCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    )
    
    # Setup optimizer
    optimizer = AMADGRADOptimizer(
        model.parameters(),
        lr=config['optimizer']['lr'],
        beta_values=config['optimizer']['beta_values'],
        eps=config['optimizer']['eps']
    )
    
    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch,
            config['training']['log_interval']
        )
        test_metrics = evaluate(model, test_loader)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_metrics["loss"]:.4f}, '
              f'Train Accuracy: {train_metrics["accuracy"]:.2f}%')
        print(f'Test Loss: {test_metrics["test_loss"]:.4f}, '
              f'Test Accuracy: {test_metrics["test_accuracy"]:.2f}%')

if __name__ == '__main__':
    # Add project root to Python path when running directly
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
