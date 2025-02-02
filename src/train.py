import torch
import torch.nn as nn
from typing import Tuple

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector

def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    num_epochs: int,
    learning_rate: float
) -> Tuple[nn.Module, list]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_data)
        target = torch.mean(train_data, dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, losses
