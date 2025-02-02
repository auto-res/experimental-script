import torch
import numpy as np

def generate_sample_data(batch_size: int, seq_len: int, input_dim: int):
    return torch.randn(batch_size, seq_len, input_dim)

def prepare_data(batch_size: int = 32, seq_len: int = 10, input_dim: int = 768):
    return generate_sample_data(batch_size, seq_len, input_dim)
