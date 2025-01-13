import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.gated_pooling import LearnableGatedPooling

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Train the model using the provided data loaders.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on (cuda/cpu)
    
    Returns:
        trained_model: The trained model
    """
    model = model.to(device)
    criterion = nn.MSELoss()  # Using MSE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"Starting training for {num_epochs} epochs")
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
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, os.path.join('models', 'best_model.pt'))
    
    return model

def save_model(model, path):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")
