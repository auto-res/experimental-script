import torch
import torch.nn as nn

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
        output = torch.nn.Linear(self.weights.shape[0], 1)(pooled_vector)
        return output

def train_model(train_loader, input_dim, seq_len, epochs=10):
    model = LearnableGatedPooling(input_dim, seq_len)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
    
    return model
