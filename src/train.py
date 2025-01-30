import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
