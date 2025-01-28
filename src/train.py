import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_data

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled = torch.mean(gated_x, dim=1)
        return self.classifier(pooled)

def train_model():
    train_data, train_labels = load_data()
    input_dim = 768
    seq_len = 10
    model = LearnableGatedPooling(input_dim, seq_len)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(train_data).squeeze()
        loss = criterion(outputs, train_labels.float())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    return model
