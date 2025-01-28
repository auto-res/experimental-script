import torch
import torch.nn as nn

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        weighted_x = x * self.weights.unsqueeze(0).unsqueeze(0)
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(-1)
        gated_x = weighted_x * gate_values.unsqueeze(-1)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector
