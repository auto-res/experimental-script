import torch
import torch.nn as nn

def evaluate_model(model: nn.Module, test_data: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        target = torch.mean(test_data, dim=1)
        mse_loss = torch.nn.functional.mse_loss(output, target)
    return mse_loss.item()
