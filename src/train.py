import torch
from .models import LearnableGatedPooling

def train_model(data, input_dim, seq_len):
    """Train a minimal model using LearnableGatedPooling."""
    model = LearnableGatedPooling(input_dim, seq_len)
    return model(data)
