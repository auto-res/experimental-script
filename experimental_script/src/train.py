import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 100
) -> float:
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                if writer is not None:
                    writer.add_scalar(
                        'train/batch_loss',
                        loss.item(),
                        epoch * len(train_loader) + batch_idx
                    )
                pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return avg_loss
