import torch
import numpy as np

def load_data(data_path):
    batch_size = 32
    seq_len = 10
    input_dim = 768
    return torch.randn(batch_size, seq_len, input_dim)

def prepare_data(data, batch_size=32):
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
