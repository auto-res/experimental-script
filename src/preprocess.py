import torch
import numpy as np

def generate_sample_data(batch_size=32, seq_len=10, input_dim=768):
    return torch.randn(batch_size, seq_len, input_dim)
