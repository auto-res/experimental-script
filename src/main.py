import torch
import logging
import argparse
from preprocess import prepare_data
from train import LearnableGatedPooling, train_model
from evaluate import evaluate_model, analyze_weights

def main(config):
    """
    Main function to run the experiment
    
    Args:
        config: Dictionary containing experiment configuration
    """
    # Set up logging
    logging.basicConfig(filename='logs/logs.txt', level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting experiment with Learnable Gated Pooling")
    logging.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Generate dummy data for demonstration
    # In practice, replace this with your actual data loading logic
    dummy_sequences = [
        torch.randn(seq_len, config['input_dim']) 
        for seq_len in torch.randint(5, config['seq_len'], (100,))
    ]
    
    # Prepare data
    train_loader, val_loader = prepare_data(
        dummy_sequences,
        config['seq_len'],
        config['input_dim'],
        config['batch_size']
    )
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=config['input_dim'],
        seq_len=config['seq_len']
    )
    
    # Train model
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=trained_model,
        test_loader=val_loader,  # Using validation set as test set for demonstration
        device=device
    )
    
    # Analyze learned parameters
    weight_analysis = analyze_weights(trained_model)
    
    # Save model
    torch.save(trained_model.state_dict(), 'models/learnable_gated_pooling.pth')
    logging.info("Model saved successfully")
    
    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Learnable Gated Pooling experiment')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    # In practice, replace this with your config loading logic
    config = {
        'input_dim': 768,  # Example: BERT embedding dimension
        'seq_len': 10,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    main(config)
