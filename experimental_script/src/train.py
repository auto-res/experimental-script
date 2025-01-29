import torch
import torch.nn as nn
from typing import Tuple, List
from ..config.model_config import ModelConfig

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim: int):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector

def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[nn.Module, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(ModelConfig.epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{ModelConfig.epochs}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), ModelConfig.model_save_path)
    return model, losses
