# Scripts for data preprocessing.
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import logging

class SequenceDataset(Dataset):
    def __init__(self, sequences: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Custom Dataset for sequence data
        
        Args:
            sequences: List of input sequences
            targets: List of target values
        """
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepare_data(
    raw_sequences: List[np.ndarray],
    raw_targets: List[np.ndarray],
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for training, validation and testing
    
    Args:
        raw_sequences: List of input sequences
        raw_targets: List of target values
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        Tuple of DataLoaders for train, validation and test sets
    """
    # Convert to torch tensors
    sequences = [torch.FloatTensor(seq) for seq in raw_sequences]
    targets = [torch.FloatTensor(target) for target in raw_targets]
    
    # Split indices
    n_samples = len(sequences)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = SequenceDataset(
        [sequences[i] for i in train_indices],
        [targets[i] for i in train_indices]
    )
    val_dataset = SequenceDataset(
        [sequences[i] for i in val_indices],
        [targets[i] for i in val_indices]
    )
    test_dataset = SequenceDataset(
        [sequences[i] for i in test_indices],
        [targets[i] for i in test_indices]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logging.info(f"Data split: Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_loader, val_loader, test_loader
