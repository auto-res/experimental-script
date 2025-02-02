from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.learnable_gated_pooling import LearnableGatedPooling

def train_model(
    model: LearnableGatedPooling,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int
) -> Dict[str, float]:
    model.train()
    train_loss = 0.0
    
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return {'train_loss': train_loss / len(train_loader)}
