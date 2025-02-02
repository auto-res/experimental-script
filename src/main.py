import torch
from preprocess import prepare_data
from train import LearnableGatedPooling, train_model
from evaluate import evaluate_model

def main():
    batch_size = 32
    seq_len = 10
    input_dim = 768
    num_epochs = 10

    print("Preparing data...")
    data = prepare_data(batch_size, seq_len, input_dim)
    
    print("Initializing model...")
    model = LearnableGatedPooling(input_dim, seq_len)
    
    print("Training model...")
    model, losses = train_model(model, data, num_epochs)
    
    print("\nTraining losses:")
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch + 1}: {loss:.6f}")
    
    print("\nEvaluating model...")
    test_data = prepare_data(batch_size, seq_len, input_dim)
    test_loss = evaluate_model(model, test_data)
    print(f"Test Loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
