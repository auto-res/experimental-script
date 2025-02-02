import torch
from pathlib import Path
from typing import Dict

from models.learnable_gated_pooling import LearnableGatedPooling
from preprocess import generate_sample_data

def evaluate_model(model: LearnableGatedPooling) -> Dict[str, float]:
    """Evaluate the trained model."""
    model.eval()
    features, targets = generate_sample_data()
    
    with torch.no_grad():
        outputs = model(features)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()
    
    return {'accuracy': accuracy}
