import torch
from train import LearnableGatedPooling

# Test shape and basic functionality
input_dim, seq_len, batch_size = 768, 10, 32
model = LearnableGatedPooling(input_dim, seq_len)
x = torch.randn(batch_size, seq_len, input_dim)
output = model(x)

# Verify output shape
print(f'Output shape: {output.shape}')

# Verify gate values are between 0 and 1
gate_values = torch.sigmoid(model.gate_linear(x))
print(f'Gate values range: [{gate_values.min().item():.3f}, {gate_values.max().item():.3f}]')

# Verify weights are being applied
print(f'Learnable weights range: [{model.weights.min().item():.3f}, {model.weights.max().item():.3f}]')
