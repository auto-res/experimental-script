import logging
import os
import torch
import torch.nn as nn
from utils.gated_pooling import LearnableGatedPooling

logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    total_samples = 0
    
    logger.info("Starting model evaluation")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    
    metrics = {
        'test_loss': avg_loss,
    }
    
    logger.info(f"Evaluation completed - Test Loss: {avg_loss:.4f}")
    
    return metrics

def load_model(path):
    """Load a trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    model = LearnableGatedPooling(input_dim=768, seq_len=10)  # Default dimensions
    model.load_state_dict(torch.load(path))
    logger.info(f"Model loaded from {path}")
    
    return model
