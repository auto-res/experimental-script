import torch
import torch.optim as optim
from models import LearnableGatedPooling

def train_model(model, num_epochs=5, batch_size=32, seq_len=10, input_dim=768):
    optimizer = optim.Adam(model.parameters())
    model.train()
    
    for epoch in range(num_epochs):
        data = torch.randn(batch_size, seq_len, input_dim)
        optimizer.zero_grad()
        
        output = model(data)
        loss = torch.mean((output - torch.mean(data, dim=1)) ** 2)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model
