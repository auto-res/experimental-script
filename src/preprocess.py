# Scripts for data preprocessing.
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

__all__ = ['load_and_preprocess_data']

def load_and_preprocess_data(data_path, input_dim=768, seq_len=10):
    """
    Load and preprocess data for the Learnable Gated Pooling model.
    For demonstration, we create synthetic data.
    
    Args:
        data_path (str): Path to data directory
        input_dim (int): Dimension of input vectors
        seq_len (int): Length of input sequences
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Create synthetic data for demonstration
    num_samples = 1000
    X_train = torch.randn(num_samples, seq_len, input_dim)
    y_train = torch.randint(0, 2, (num_samples,))  # Binary labels
    
    # Split into train and validation
    split_idx = int(0.8 * num_samples)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Create data loaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
