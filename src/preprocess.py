import torch
import numpy as np

def load_and_preprocess_data(data_path, seq_len, input_dim):
    batch_size = 32
    X = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, 2, (batch_size,))
    return X, y

def create_data_loader(X, y, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
