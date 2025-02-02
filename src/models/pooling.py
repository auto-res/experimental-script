import torch
import torch.nn as nn
from typing import Tuple

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        return torch.mean(gated_x, dim=1)
