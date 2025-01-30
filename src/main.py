import torch
from preprocess import load_and_preprocess_data, create_data_loaders
from train import LearnableGatedPooling, train_model
from evaluate import evaluate_model

def main():
    input_dim = 768
    seq_len = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    
    x, y = load_and_preprocess_data("data/", seq_len, input_dim)
    train_loader, test_loader = create_data_loaders(x, y, batch_size)
    
    model = LearnableGatedPooling(input_dim, seq_len)
    history = train_model(model, train_loader, num_epochs, learning_rate)
    
    metrics = evaluate_model(model, test_loader)
    print(f'Test Loss: {metrics["test_loss"]:.4f}')

if __name__ == "__main__":
    main()
