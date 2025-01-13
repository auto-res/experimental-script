import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """Custom dataset for handling sequence data."""
    def __init__(self, sequences, targets=None, seq_len=10, input_dim=768):
        """
        Initialize the dataset.
        
        Args:
            sequences: Input sequences of shape (num_sequences, seq_len, input_dim)
            targets: Target values for sequences (if None, uses sequences as targets)
            seq_len: Length of sequences
            input_dim: Dimension of input vectors
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        if targets is None:
            # If no targets provided, use sequences themselves (e.g., for autoencoding)
            self.targets = self.sequences
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Validate dimensions
        if self.sequences.shape[1] != seq_len or self.sequences.shape[2] != input_dim:
            raise ValueError(f"Expected sequences of shape (N, {seq_len}, {input_dim}), "
                           f"got {self.sequences.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepare_data(data_path, seq_len=10, input_dim=768, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to the data directory
        seq_len: Length of sequences
        input_dim: Dimension of input vectors
        batch_size: Batch size for DataLoader
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load sequences from data directory
    # Assuming data is stored as numpy arrays
    sequences_path = os.path.join(data_path, 'sequences.npy')
    targets_path = os.path.join(data_path, 'targets.npy')
    
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"Sequences file not found at {sequences_path}")
    
    sequences = np.load(sequences_path)
    if os.path.exists(targets_path):
        targets = np.load(targets_path)
    else:
        logger.info("No targets file found, using sequences as targets")
        targets = sequences
    
    # Split data into train, validation, and test sets
    train_seq, temp_seq, train_targets, temp_targets = train_test_split(
        sequences, targets, test_size=(val_split + test_split), random_state=42
    )
    
    val_seq, test_seq, val_targets, test_targets = train_test_split(
        temp_seq, temp_targets, test_size=test_split/(val_split + test_split), random_state=42
    )
    
    # Create datasets
    train_dataset = CustomDataset(train_seq, train_targets, seq_len, input_dim)
    val_dataset = CustomDataset(val_seq, val_targets, seq_len, input_dim)
    test_dataset = CustomDataset(test_seq, test_targets, seq_len, input_dim)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Data split: Train={len(train_seq)}, Val={len(val_seq)}, Test={len(test_seq)}")
    
    return train_loader, val_loader, test_loader
