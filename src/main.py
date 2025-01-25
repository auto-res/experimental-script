import logging
import os
import torch
from datetime import datetime
from utils.gated_pooling import LearnableGatedPooling
from preprocess import prepare_data
from train import train_model
from evaluate import evaluate_model

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'logs.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function to run the experimental pipeline."""
    logger = setup_logging()
    logger.info("Starting experimental pipeline")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Configuration
        config = {
            'seq_len': 10,
            'input_dim': 768,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'data_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        }
        
        # Load and preprocess data
        logger.info("Preparing data...")
        train_loader, val_loader, test_loader = prepare_data(
            config['data_path'],
            config['seq_len'],
            config['input_dim'],
            config['batch_size']
        )
        
        # Initialize model
        logger.info("Initializing LearnableGatedPooling model...")
        model = LearnableGatedPooling(
            input_dim=config['input_dim'],
            seq_len=config['seq_len']
        )
        
        # Train model
        logger.info("Starting model training...")
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            config['num_epochs'],
            config['learning_rate'],
            device
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(trained_model, test_loader, device)
        
        logger.info(f"Final test metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in experimental pipeline: {str(e)}", exc_info=True)
        raise
    
    logger.info("Experimental pipeline completed")

if __name__ == "__main__":
    main()
    print("Done!")
