# Scripts for data preprocessing.

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import os

class SequenceDataset(Dataset):
    """Dataset class for handling sequence data for LearnableGatedPooling."""
    
    def __init__(self, sequences: List[torch.Tensor], seq_len: int, input_dim: int):
        """Initialize the dataset.
        
        Args:
            sequences (List[torch.Tensor]): List of input sequences
            seq_len (int): Maximum sequence length (will pad/truncate to this length)
            input_dim (int): Dimension of input vectors
        """
        self.sequences = sequences
        self.seq_len = seq_len
        self.input_dim = input_dim
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sequence from the dataset.
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            torch.Tensor: Padded/truncated sequence of shape (seq_len, input_dim)
        """
        sequence = self.sequences[idx]
        
        # Pad or truncate sequence to seq_len
        if sequence.size(0) > self.seq_len:
            sequence = sequence[:self.seq_len]
        elif sequence.size(0) < self.seq_len:
            padding = torch.zeros(self.seq_len - sequence.size(0), self.input_dim)
            sequence = torch.cat([sequence, padding], dim=0)
            
        return sequence

def preprocess_sequences(
    raw_sequences: List[torch.Tensor],
    seq_len: int,
    input_dim: int,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Preprocess sequences and create data loaders.
    
    Args:
        raw_sequences (List[torch.Tensor]): List of raw input sequences
        seq_len (int): Maximum sequence length
        input_dim (int): Dimension of input vectors
        batch_size (int, optional): Batch size for data loader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        
    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: Train and optional validation data loaders
    """
    # Create dataset
    dataset = SequenceDataset(raw_sequences, seq_len, input_dim)
    
    # Split into train/validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

def load_data(data_dir: str) -> List[torch.Tensor]:
    """Load raw sequence data from files.
    
    Args:
        data_dir (str): Directory containing the sequence data files
        
    Returns:
        List[torch.Tensor]: List of raw sequences
    """
    # This is a placeholder implementation
    # In a real application, this would load actual sequence data from files
    # For demonstration, we'll create some random sequences
    sequences = []
    for _ in range(100):  # Create 100 random sequences
        seq_len = np.random.randint(5, 15)  # Random sequence length between 5 and 15
        sequence = torch.randn(seq_len, 768)  # Random sequence with BERT-like dimensions
        sequences.append(sequence)
    return sequences

if __name__ == "__main__":
    # Test the preprocessing pipeline
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # Load and preprocess data
    raw_sequences = load_data(data_dir)
    train_loader, val_loader = preprocess_sequences(
        raw_sequences,
        seq_len=10,
        input_dim=768,
        batch_size=32
    )
    
    # Verify the data loaders
    for batch in train_loader:
        assert batch.shape == (batch.size(0), 10, 768), \
            f"Expected shape (N, 10, 768), but got {batch.shape}"
        break
    
    print("Preprocessing pipeline test passed successfully!")
