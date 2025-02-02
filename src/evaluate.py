from typing import Dict
import torch
from torch.utils.data import DataLoader
from models.learnable_gated_pooling import LearnableGatedPooling

def evaluate_model(
    model: LearnableGatedPooling,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return {'test_loss': total_loss / len(test_loader)}
