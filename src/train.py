import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocess import load_data

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim):
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(-1)
        gated_x = weighted_x * gate_values.unsqueeze(-1)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector

def train_model():
    X, y = load_data()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = LearnableGatedPooling(input_dim=768)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        outputs = model(X_tensor)
        logits = model.classifier(outputs)
        loss = criterion(logits, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training complete!")
