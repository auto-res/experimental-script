from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model: torch.nn.Module,
            loader: DataLoader,
            device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": 100. * correct / len(loader.dataset)
    }
