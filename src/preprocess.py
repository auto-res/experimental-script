# Scripts for data preprocessing.

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    """
    Custom Dataset class for sequence data.
    """
    def __init__(self, sequences, targets):
        """
        Args:
            sequences (np.ndarray): Input sequences of shape (n_samples, seq_len, input_dim)
            targets (np.ndarray): Target values of shape (n_samples, output_dim)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepare_data(data_path, seq_len, input_dim, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare and split data into train, validation, and test sets.
    
    Args:
        data_path (str): Path to raw data
        seq_len (int): Length of input sequences
        input_dim (int): Dimension of input features
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
    
    Returns:
        tuple: Train, validation, and test DataLoader objects
    """
    # Load and preprocess data (implement based on data format)
    # This is a placeholder implementation
    n_samples = 1000  # Example size
    sequences = np.random.randn(n_samples, seq_len, input_dim)
    targets = np.random.randn(n_samples, input_dim)  # Example target
    
    # Split data
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    
    val_sequences = sequences[train_size:train_size+val_size]
    val_targets = targets[train_size:train_size+val_size]
    
    test_sequences = sequences[train_size+val_size:]
    test_targets = targets[train_size+val_size:]
    
    # Create datasets
    train_dataset = SequenceDataset(train_sequences, train_targets)
    val_dataset = SequenceDataset(val_sequences, val_targets)
    test_dataset = SequenceDataset(test_sequences, test_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader
