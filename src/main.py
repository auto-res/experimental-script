# Main script for running the experiment

import torch
import torch.nn as nn
import logging
import json
from pathlib import Path

from models.learnable_gated_pooling import LearnableGatedPooling
from preprocess import prepare_data
from train import train_model, save_model
from evaluate import evaluate_model, load_model

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)

def main():
    # Set up logging
    setup_logging()
    
    # Load configuration
    config_path = Path("config/experiment_config.json")
    config = load_config(config_path)
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        data_path=config['data_path'],
        seq_len=config['seq_len'],
        input_dim=config['input_dim'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=config['input_dim'],
        seq_len=config['seq_len']
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Save model
    model_path = Path("models") / config['model_save_name']
    save_model(trained_model, model_path)
    
    # Evaluate model
    criterion = nn.MSELoss()
    metrics = evaluate_model(trained_model, test_loader, criterion)
    
    # Log results
    logging.info("Training completed. Test metrics:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
