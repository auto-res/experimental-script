# Scripts for evaluation.

import torch
import numpy as np
from torch.utils.data import DataLoader
import logging
from models.learnable_gated_pooling import LearnableGatedPooling

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the trained model on test data.
    
    Args:
        model (LearnableGatedPooling): Trained model
        test_loader (DataLoader): DataLoader for test data
        criterion: Loss function
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate additional metrics (can be customized based on task)
    metrics = {
        'test_loss': avg_loss,
        'mse': np.mean((all_predictions - all_targets) ** 2),
        'mae': np.mean(np.abs(all_predictions - all_targets))
    }
    
    return metrics

def load_model(model, path):
    """
    Load a trained model from disk.
    
    Args:
        model: PyTorch model instance
        path (str): Path to the saved model
    
    Returns:
        model: Loaded model
    """
    model.load_state_dict(torch.load(path))
    return model
