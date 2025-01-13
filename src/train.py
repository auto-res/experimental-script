# Scripts for training models.
import torch
import torch.nn as nn
import torch.optim as optim
from models.learnable_gated_pooling import LearnableGatedPooling
import logging
import os

__all__ = ['train_model']

def train_model(train_loader, val_loader, input_dim=768, seq_len=10, num_epochs=10):
    """
    Train the Learnable Gated Pooling model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim (int): Dimension of input vectors
        seq_len (int): Length of input sequences
        num_epochs (int): Number of training epochs
    
    Returns:
        model: Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableGatedPooling(input_dim, seq_len).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join('logs', 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_sequences, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.float().to(device)
                
                outputs = model(batch_sequences)
                loss = criterion(outputs.squeeze(), batch_labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Training Loss: {train_loss:.4f}')
        logging.info(f'Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))
    
    return model
