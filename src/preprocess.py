import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(filename='logs/logs.txt', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class SequenceDataset(Dataset):
    def __init__(self, sequences, seq_len, input_dim):
        """
        Args:
            sequences: List of input sequences
            seq_len: Maximum sequence length
            input_dim: Dimension of input vectors
        """
        self.sequences = sequences
        self.seq_len = seq_len
        self.input_dim = input_dim
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Pad or truncate sequence to seq_len
        if len(sequence) < self.seq_len:
            padding = torch.zeros((self.seq_len - len(sequence), self.input_dim))
            sequence = torch.cat([sequence, padding], dim=0)
        else:
            sequence = sequence[:self.seq_len]
        return sequence

def prepare_data(raw_sequences, seq_len, input_dim, batch_size):
    """
    Prepare data for training and evaluation
    
    Args:
        raw_sequences: List of input sequences
        seq_len: Maximum sequence length
        input_dim: Dimension of input vectors
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    logging.info("Preparing data for training and validation")
    
    # Create dataset
    dataset = SequenceDataset(raw_sequences, seq_len, input_dim)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logging.info(f"Created train loader with {len(train_loader)} batches")
    logging.info(f"Created validation loader with {len(val_loader)} batches")
    
    return train_loader, val_loader
