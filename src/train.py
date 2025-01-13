# Scripts for training models.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from models.learnable_gated_pooling import LearnableGatedPooling

def train_model(model, train_loader, val_loader, config):
    """
    Train the LearnableGatedPooling model.
    
    Args:
        model (LearnableGatedPooling): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        config (dict): Training configuration parameters
    
    Returns:
        model: Trained model
        dict: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()  # Can be modified based on task requirements
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        logging.info(f'Epoch {epoch+1}/{config["epochs"]} - '
                    f'Train Loss: {avg_train_loss:.4f} - '
                    f'Val Loss: {avg_val_loss:.4f}')
    
    return model, history

def save_model(model, path):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained PyTorch model
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)
