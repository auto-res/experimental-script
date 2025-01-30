from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model

def main():
    """Run the full pipeline for Learnable Gated Pooling demonstration."""
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Train model
    train_model()
    
    # Evaluate model
    evaluate_model()

if __name__ == "__main__":
    main()
