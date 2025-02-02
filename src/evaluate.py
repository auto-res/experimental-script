import torch
import torch.nn as nn
from typing import Dict, Tuple

def evaluate_model(
    model: nn.Module,
    test_data: Tuple[torch.Tensor, torch.Tensor]
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_test, y_test = test_data
        output = model(x_test)
        mse = torch.nn.functional.mse_loss(output, y_test)
    return {'mse': mse.item()}
