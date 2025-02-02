import torch
import torch.nn as nn
from models.learnable_gated_pooling import LearnableGatedPooling
from train import train_model
from evaluate import evaluate_model
from preprocess import get_dataloaders
from config.model_config import ModelConfig

def main():
    # Load configuration
    config = ModelConfig()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=config.input_dim,
        seq_len=config.seq_len
    ).to(device)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        input_dim=config.input_dim
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train model
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config.epochs
    )
    
    # Evaluate model
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"Training metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
