import torch
from preprocess import prepare_data
from train import train_model
from evaluate import evaluate_model

def main():
    input_dim = 768
    seq_len = 10
    batch_size = 32
    
    sample_data = torch.randn(100, seq_len, input_dim)
    train_loader = prepare_data(sample_data, batch_size)
    
    model = train_model(input_dim, seq_len, train_loader)
    
    test_data = torch.randn(20, seq_len, input_dim)
    test_loader = prepare_data(test_data, batch_size)
    loss = evaluate_model(model, test_loader)
    print(f"Test Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
