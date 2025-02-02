import torch
import torch.nn as nn
from typing import Tuple
import yaml
from models.pooling import LearnableGatedPooling

def train_model(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    config: dict
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.MSELoss()
    
    x_train, y_train = train_data
    for epoch in range(config['model']['num_epochs']):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{config["model"]["num_epochs"]}], Loss: {loss.item():.4f}')
    
    return model
