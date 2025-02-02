import torch
import torch.nn as nn
from typing import Dict

def evaluate_model(
    model: nn.Module,
    test_data: torch.Tensor
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        target = torch.mean(test_data, dim=1)
        mse = nn.MSELoss()(output, target).item()
        mae = nn.L1Loss()(output, target).item()
    
    return {
        'mse': mse,
        'mae': mae
    }
