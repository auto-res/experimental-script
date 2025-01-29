import torch
import torch.nn as nn
import torch.optim as optim
from src.models import LearnableGatedPooling
from src.train import train_model
from src.evaluate import evaluate_model
from src.preprocess import create_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = 768
    seq_len = 10
    
    model = LearnableGatedPooling(input_dim, seq_len).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    sequences = torch.randn(100, seq_len, input_dim)
    labels = torch.randn(100, input_dim)
    
    train_loader = create_dataloaders(sequences, labels)
    
    model = train_model(model, train_loader, optimizer, criterion, device)
    loss = evaluate_model(model, train_loader, criterion, device)
    print(f"Final loss: {loss:.4f}")

if __name__ == "__main__":
    main()
