import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)  # Linear layer for gating

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)  # (batch_size, seq_len)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)  # Average pooling
        return pooled_vector

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device='cuda'):
    """
    Train the LearnableGatedPooling model
    
    Args:
        model: LearnableGatedPooling model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    logging.info(f"Starting training for {num_epochs} epochs")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            # For demonstration, we use reconstruction loss
            loss = criterion(output, torch.mean(batch, dim=1))
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, torch.mean(batch, dim=1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Training Loss: {avg_train_loss:.4f}')
        logging.info(f'Validation Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses
