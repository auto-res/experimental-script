import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import os
import sys

import os
import sys

from utils.optimizers import CustomOptimizer
from preprocess import get_mnist_loaders
from train import SimpleConvNet, train_epoch
from evaluate import evaluate
from config.train_config import TrainConfig

def main():
    config = TrainConfig()
    
    # Create log directory
    log_dir = os.path.join(
        'logs',
        f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    writer = SummaryWriter(log_dir)
    
    # Setup data
    train_loader, test_loader = get_mnist_loaders(
        config.batch_size,
        config.num_workers
    )
    
    # Setup model and optimizer
    model = SimpleConvNet().to(config.device)
    optimizer = CustomOptimizer(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
        adaptive_rate=config.adaptive_rate
    )
    
    # Training loop
    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            epoch,
            writer,
            config.log_interval
        )
        
        metrics = evaluate(model, test_loader, config.device)
        
        print(
            f'Epoch {epoch}: '
            f'Train Loss: {train_loss:.4f}, '
            f'Test Loss: {metrics["test_loss"]:.4f}, '
            f'Accuracy: {metrics["accuracy"]:.4f}'
        )
        
        writer.add_scalar('test/loss', metrics['test_loss'], epoch)
        writer.add_scalar('test/accuracy', metrics['accuracy'], epoch)
    
    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join('models', 'final_model.pt')
    )

if __name__ == '__main__':
    main()
