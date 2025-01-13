import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Optional, Tuple

from models import LearnableGatedPooling
from preprocess import preprocess_sequences, load_data

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models"
) -> Tuple[list, list]:
    """Train the LearnableGatedPooling model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (Optional[DataLoader], optional): DataLoader for validation data
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        device (str, optional): Device to train on. Defaults to "cuda" if available.
        save_dir (str, optional): Directory to save model checkpoints. Defaults to "models".
        
    Returns:
        Tuple[list, list]: Lists of training and validation losses
    """
    model = model.to(device)
    criterion = nn.MSELoss()  # Using MSE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for batch_idx, sequences in enumerate(train_loader):
            sequences = sequences.to(device)
            
            # Forward pass
            outputs = model(sequences)
            
            # For this example, we'll use the original sequences as targets
            # In a real application, you would have actual target values
            target = torch.mean(sequences, dim=1)  # Simple pooling as target
            
            # Compute loss
            loss = criterion(outputs, target)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sequences in val_loader:
                    sequences = sequences.to(device)
                    outputs = model(sequences)
                    target = torch.mean(sequences, dim=1)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Training Loss: {avg_train_loss:.4f} "
                  f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 
                         os.path.join(save_dir, "best_model.pth"))
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Training Loss: {avg_train_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(save_dir, "final_model.pth"))
    
    return train_losses, val_losses

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    
    # Model parameters
    input_dim = 768  # Example: BERT embedding dimension
    seq_len = 10
    
    # Load and preprocess data
    raw_sequences = load_data(data_dir)
    train_loader, val_loader = preprocess_sequences(
        raw_sequences,
        seq_len=seq_len,
        input_dim=input_dim,
        batch_size=32
    )
    
    # Initialize model
    model = LearnableGatedPooling(input_dim, seq_len)
    
    # Train model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        save_dir=save_dir
    )
    
    print("Training completed successfully!")
