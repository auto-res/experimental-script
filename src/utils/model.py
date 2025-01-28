import torch
import torch.nn as nn

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        weighted_x = x * self.weights  # weights broadcast automatically
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(-1)  # [batch_size, seq_len]
        gated_x = weighted_x * gate_values.unsqueeze(-1)  # [batch_size, seq_len, input_dim]
        pooled_vector = torch.mean(gated_x, dim=1)  # [batch_size, input_dim]
        return pooled_vector
