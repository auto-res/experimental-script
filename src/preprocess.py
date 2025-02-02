from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, input_dim: int):
        self.data = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randn(num_samples, input_dim)  # Dummy labels
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def get_dataloaders(
    batch_size: int = 32,
    num_samples: int = 1000,
    seq_len: int = 10,
    input_dim: int = 768
) -> Tuple[DataLoader, DataLoader]:
    dataset = DummyDataset(num_samples, seq_len, input_dim)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
