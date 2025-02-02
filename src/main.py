import torch
from models import LearnableGatedPooling
from preprocess import generate_sample_data
from train import train_model
from evaluate import evaluate_model

def main():
    print("=== Learnable Gated Pooling Experiment ===")
    
    # Model parameters
    input_dim = 768
    seq_len = 10
    batch_size = 32
    
    print(f"\nParameters:")
    print(f"Input dimension: {input_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    
    # Initialize model
    model = LearnableGatedPooling(input_dim, seq_len)
    print("\nModel architecture:")
    print(model)
    
    # Training
    print("\nTraining:")
    model = train_model(model, num_epochs=5, batch_size=batch_size, seq_len=seq_len, input_dim=input_dim)
    
    # Evaluation
    print("\nEvaluation:")
    test_data = generate_sample_data(batch_size=batch_size, seq_len=seq_len, input_dim=input_dim)
    test_loss = evaluate_model(model, test_data)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
