import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim: int, seq_len: int):
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
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3,
) -> Dict[str, List[float]]:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {"loss": []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)
        
    return history
