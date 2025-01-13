import os
from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model
import logging

def main():
    """
    Main function to run the Learnable Gated Pooling experiment.
    """
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=os.path.join('logs', 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Configuration
    input_dim = 768  # Example: BERT embedding dimension
    seq_len = 10
    num_epochs = 10
    data_path = 'data'
    
    logging.info("Starting Learnable Gated Pooling experiment")
    
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        train_loader, val_loader = load_and_preprocess_data(
            data_path, input_dim, seq_len
        )
        
        # Train model
        logging.info("Training model...")
        model = train_model(
            train_loader, val_loader,
            input_dim=input_dim,
            seq_len=seq_len,
            num_epochs=num_epochs
        )
        
        # Evaluate model
        logging.info("Evaluating model...")
        metrics = evaluate_model(model, val_loader)
        
        logging.info("Experiment completed successfully")
        
    except Exception as e:
        logging.error(f"Error during experiment: {str(e)}")
        raise

if __name__ == "__main__":
    main()
