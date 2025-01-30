import torch
import torch.nn as nn
from typing import Dict

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return {"test_loss": avg_loss}
