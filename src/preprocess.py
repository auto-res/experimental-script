import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(X, y, batch_size=32, train_ratio=0.8):
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_idx = int(train_ratio * num_samples)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
