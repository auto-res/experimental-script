import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from .optimizers.aggmo_madgrad import AggMoMADGRAD

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int
) -> Dict[str, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} '
                  f'Loss: {loss.item():.6f} '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }
