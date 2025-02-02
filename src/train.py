import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from src.optimizers.amadgrad import AMADGRADOptimizer

def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                optimizer: AMADGRADOptimizer,
                epoch: int,
                log_interval: int) -> Dict[str, float]:
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / len(train_loader.dataset)
    }
    return metrics
