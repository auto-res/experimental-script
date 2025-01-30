import torch
from typing import Tuple

def load_and_preprocess_data(data_path: str, seq_len: int, input_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 32
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, input_dim)
    return x, y

def create_data_loaders(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = torch.utils.data.TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
