import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

def evaluate(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': accuracy
    }
    return metrics
