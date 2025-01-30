import torch
from .models import LearnableGatedPooling
from .preprocess import load_data, prepare_data
from .train import train_model
from .evaluate import evaluate_model

def main():
    # Load and prepare data
    data = load_data('data/sample_data.pt')
    train_loader = prepare_data(data, batch_size=32)
    
    # Initialize model
    input_dim = 768
    seq_len = 10
    model = LearnableGatedPooling(input_dim, seq_len)
    
    # Train model
    train_model(model, train_loader, num_epochs=10)
    
    # Evaluate model
    eval_results = evaluate_model(model, train_loader)
    print(f"Evaluation Results: {eval_results}")

if __name__ == "__main__":
    main()
