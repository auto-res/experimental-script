import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from .preprocess import get_mnist_loaders
from .train import train_epoch
from .evaluate import evaluate
from .optimizers.hybrid_optimizer import HybridOptimizer

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config_path = Path('config/default_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_mnist_loaders(config['training']['batch_size'])
    
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1600, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = HybridOptimizer(
        model.parameters(),
        lr=config['optimizer']['lr'],
        betas=config['optimizer']['betas'],
        weight_decay=config['optimizer']['weight_decay'],
        eps=config['optimizer']['eps']
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config['training']['epochs']):
        logging.info(f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, config['training']['log_interval']
        )
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        logging.info(
            f'Epoch {epoch+1} - '
            f'Train Loss: {train_metrics["loss"]:.6f} '
            f'Train Acc: {train_metrics["accuracy"]:.2f}% '
            f'Test Loss: {test_metrics["loss"]:.6f} '
            f'Test Acc: {test_metrics["accuracy"]:.2f}%'
        )

if __name__ == '__main__':
    main()
