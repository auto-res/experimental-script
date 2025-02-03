import yaml
import torch
from pathlib import Path
from src.models import SimpleCNN
from src.optimizers import AdvancedOptimizer
from src.preprocess import get_dataloaders
from src.train import train_epoch
from src.evaluate import evaluate

def main():
    # Load config
    config_path = Path("config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Setup optimizer
    optimizer_config = config['training']['optimizer'].copy()
    optimizer_config.pop('name', None)  # Remove name parameter if it exists
    optimizer = AdvancedOptimizer(
        model.parameters(),
        lr=config['training']['learning_rate'],
        **optimizer_config
    )
    
    # Get data
    train_loader, val_loader = get_dataloaders(config)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Accuracy: {val_metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
