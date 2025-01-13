import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Dict, Optional
import json

from models import LearnableGatedPooling
from preprocess import preprocess_sequences, load_data

def load_model(
    model: nn.Module,
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """Load a trained model from checkpoint.
    
    Args:
        model (nn.Module): Initialized model instance
        model_path (str): Path to the model checkpoint
        device (str, optional): Device to load model on. Defaults to "cuda" if available.
        
    Returns:
        nn.Module: Loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Evaluate the model on test data.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for test data
        device (str, optional): Device to evaluate on. Defaults to "cuda" if available.
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            
            # Forward pass
            outputs = model(sequences)
            
            # For this example, we'll use the original sequences as targets
            # In a real application, you would have actual target values
            target = torch.mean(sequences, dim=1)
            
            # Calculate metrics
            mse = torch.mean((outputs - target) ** 2)
            mae = torch.mean(torch.abs(outputs - target))
            
            # Update running statistics
            batch_size = sequences.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size
    
    # Calculate final metrics
    metrics = {
        "mse": total_mse / total_samples,
        "mae": total_mae / total_samples,
        "rmse": (total_mse / total_samples) ** 0.5
    }
    
    return metrics

def save_metrics(metrics: Dict[str, float], save_path: str):
    """Save evaluation metrics to a JSON file.
    
    Args:
        metrics (Dict[str, float]): Dictionary of evaluation metrics
        save_path (str): Path to save the metrics file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    
    # Model parameters
    input_dim = 768  # Example: BERT embedding dimension
    seq_len = 10
    
    # Load and preprocess test data
    raw_sequences = load_data(data_dir)  # In practice, this would be test data
    _, test_loader = preprocess_sequences(
        raw_sequences,
        seq_len=seq_len,
        input_dim=input_dim,
        batch_size=32
    )
    
    # Initialize and load trained model
    model = LearnableGatedPooling(input_dim, seq_len)
    model_path = os.path.join(models_dir, "best_model.pth")
    if os.path.exists(model_path):
        model = load_model(model, model_path)
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader)
        
        # Save and display metrics
        metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
        save_metrics(metrics, metrics_path)
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
    else:
        print(f"No trained model found at {model_path}")
        print("Please train the model first using Train.py")
