import torch
from models import LearnableGatedPooling

def test_learnable_gated_pooling():
    # Test data
    input_dim = 768
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim)

    # Initialize model
    model = LearnableGatedPooling(input_dim, seq_len)

    # Forward pass
    output = model(x)

    # Verify output shape
    expected_shape = (batch_size, input_dim)
    assert output.shape == expected_shape, f'Expected shape {expected_shape}, got {output.shape}'

    print('Test passed successfully! Output shape:', output.shape)

if __name__ == "__main__":
    test_learnable_gated_pooling()
