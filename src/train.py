import torch
import torch.nn as nn
from utils.model import LearnableGatedPooling

def train_model(input_dim, seq_len, train_data, num_epochs=10, learning_rate=0.001):
    model = LearnableGatedPooling(input_dim, seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_data:
            optimizer.zero_grad()
            output = model(batch[0])
            loss = criterion(output, torch.mean(batch[0], dim=1))
            loss.backward()
            optimizer.step()

    return model
