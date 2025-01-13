# Scripts for training models.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LearnableGatedPooling
import logging
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    """
    Train the LearnableGatedPooling model
    
    Args:
        model: LearnableGatedPooling model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        save_dir: Directory to save model checkpoints
    """
    logging.info("Starting training...")
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        
        logging.info(f'Epoch: {epoch+1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
    logging.info("Training completed!")
    return model
