import torch
import torch.nn as nn
from typing import Dict

def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(x)
        mse = torch.mean((output - y) ** 2)
        mae = torch.mean(torch.abs(output - y))
    return {"mse": mse.item(), "mae": mae.item()}
