import os
import json
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from models import LearnableGatedPooling
from preprocess import load_data, preprocess_sequences
from Train import train_model
from Evaluate import load_model, evaluate_model, save_metrics

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {save_path}")

def run_experiment(config: Dict[str, Any]):
    """Run the complete experiment pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing experiment parameters
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['models_dir'], exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    raw_sequences = load_data(config['data_dir'])
    train_loader, val_loader = preprocess_sequences(
        raw_sequences,
        seq_len=config['seq_len'],
        input_dim=config['input_dim'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = LearnableGatedPooling(
        input_dim=config['input_dim'],
        seq_len=config['seq_len']
    )
    
    # Training phase
    print("\nStarting training phase...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device,
        save_dir=config['models_dir']
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    history_path = os.path.join(config['models_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_path}")
    
    # Evaluation phase
    print("\nStarting evaluation phase...")
    best_model_path = os.path.join(config['models_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        # Load best model
        model = load_model(model, best_model_path, device)
        
        # Evaluate on validation set
        metrics = evaluate_model(model, val_loader, device)
        
        # Save evaluation metrics
        metrics_path = os.path.join(config['models_dir'], 'evaluation_metrics.json')
        save_metrics(metrics, metrics_path)
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
    else:
        print(f"No trained model found at {best_model_path}")

def main():
    """Main entry point for running the experiment."""
    parser = argparse.ArgumentParser(description='Run LearnableGatedPooling experiment')
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Set up default configuration
    config = {
        'data_dir': 'data',
        'models_dir': 'models',
        'input_dim': 768,  # Example: BERT embedding dimension
        'seq_len': 10,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    # Load configuration from file if it exists
    if os.path.exists(args.config):
        config.update(load_config(args.config))
    else:
        # Save default configuration
        os.makedirs('config', exist_ok=True)
        save_config(config, args.config)
        print(f"Default configuration saved to {args.config}")
    
    # Convert relative paths to absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['data_dir'] = os.path.join(base_dir, config['data_dir'])
    config['models_dir'] = os.path.join(base_dir, config['models_dir'])
    
    # Run the experiment
    run_experiment(config)

if __name__ == "__main__":
    main()
