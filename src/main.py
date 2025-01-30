import torch
import numpy as np
from train import LearnableGatedPooling, train_model
from evaluate import evaluate_model
from preprocess import prepare_data

def main():
    input_dim = 768
    seq_len = 10
    batch_size = 32
    num_samples = 1000
    
    X = np.random.randn(num_samples, seq_len, input_dim)
    y = np.random.randn(num_samples, input_dim)
    
    train_loader, val_loader = prepare_data(X, y, batch_size=batch_size)
    
    model = LearnableGatedPooling(input_dim, seq_len)
    trained_model = train_model(model, train_loader, val_loader)
    
    results = evaluate_model(trained_model, val_loader)
    print(f"MSE: {results['mse']:.4f}")

if __name__ == "__main__":
    main()
