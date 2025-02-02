import yaml
from models.pooling import LearnableGatedPooling
from preprocess import generate_synthetic_data
from train import train_model
from evaluate import evaluate_model

def main():
    # Load configuration
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = LearnableGatedPooling(input_dim=config['model']['input_dim'])
    
    # Generate synthetic data
    train_data = generate_synthetic_data(
        num_samples=100,
        seq_len=10,
        input_dim=config['model']['input_dim']
    )
    test_data = generate_synthetic_data(
        num_samples=20,
        seq_len=10,
        input_dim=config['model']['input_dim']
    )
    
    # Train model
    model = train_model(model, train_data, config)
    
    # Evaluate model
    metrics = evaluate_model(model, test_data)
    print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()
