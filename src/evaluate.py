# Scripts for evaluation.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import LearnableGatedPooling

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the LearnableGatedPooling model
    
    Args:
        model: Trained LearnableGatedPooling model instance
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    total_loss = 0
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
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    
    metrics = {
        'test_loss': avg_loss,
        'mse': mse,
        'mae': mae
    }
    
    logging.info(f"Test Results: Loss: {avg_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return metrics
