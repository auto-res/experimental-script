import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import argparse
from model import LearnableGatedPooling
from .train import train_model
from .evaluate import evaluate_model
from .preprocess import prepare_data

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main(config):
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=config.input_dim,
        seq_len=config.seq_len
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        raw_sequences=config.sequences,
        raw_targets=config.targets,
        batch_size=config.batch_size
    )
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        device=device,
        save_dir=config.model_dir
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    logging.info("Experiment completed successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Learnable Gated Pooling experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = None  # TODO: Load config from file
    
    main(config)
