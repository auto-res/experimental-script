import torch
import pytest
from src.models import LearnableGatedPooling

def test_learnable_gated_pooling():
    input_dim = 768
    seq_len = 10
    batch_size = 32
    
    model = LearnableGatedPooling(input_dim, seq_len)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    assert output.shape == (batch_size, input_dim)
    
    # Test that weights and gates are learnable
    assert isinstance(model.weights, torch.nn.Parameter)
    assert isinstance(model.gate_linear.weight, torch.nn.Parameter)
