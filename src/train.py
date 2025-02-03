from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model: torch.nn.Module,
               loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0
    correct = 0
    
    for data, target in tqdm(loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": 100. * correct / len(loader.dataset)
    }
