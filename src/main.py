import torch
from train import LearnableGatedPooling, train_model
from preprocess import generate_synthetic_data, prepare_data
from evaluate import evaluate_model

def main():
    # Configuration
    batch_size = 32
    seq_len = 10
    input_dim = 768
    num_epochs = 100
    learning_rate = 0.001

    # Generate synthetic data
    data = generate_synthetic_data(batch_size, seq_len, input_dim)
    train_data, test_data = prepare_data(data)

    # Initialize and train model
    model = LearnableGatedPooling(input_dim)
    trained_model, losses = train_model(
        model, train_data, num_epochs, learning_rate
    )

    # Evaluate model
    metrics = evaluate_model(trained_model, test_data)
    
    print(f"Training complete. Final loss: {losses[-1]:.4f}")
    print(f"Test metrics: MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()
