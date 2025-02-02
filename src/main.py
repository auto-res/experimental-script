import torch
from preprocess import load_and_preprocess_data, create_data_loader
from train import train_model
from evaluate import evaluate_model

def main():
    input_dim = 768
    seq_len = 10
    batch_size = 32
    
    X, y = load_and_preprocess_data(None, seq_len, input_dim)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_loader = create_data_loader(X_train, y_train, batch_size)
    test_loader = create_data_loader(X_test, y_test, batch_size)
    
    model = train_model(train_loader, input_dim, seq_len)
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
