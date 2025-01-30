# Scripts for data preprocessing.
import torch

def generate_dummy_data(batch_size=2, seq_len=10, input_dim=8):
    return torch.randn(batch_size, seq_len, input_dim)
