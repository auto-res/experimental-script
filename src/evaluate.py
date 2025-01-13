import torch
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained LearnableGatedPooling model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    logging.info("Starting model evaluation")
    
    model = model.to(device)
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            batch = batch.to(device)
            output = model(batch)
            target = torch.mean(batch, dim=1)  # For demonstration
            
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_outputs)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'rmse': rmse
    }
    
    logging.info("Evaluation Results:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    
    return metrics

def analyze_weights(model):
    """
    Analyze the learned weights and gates
    
    Args:
        model: Trained LearnableGatedPooling model
    """
    logging.info("Analyzing model weights and gates")
    
    # Analyze learned weights
    weights = model.weights.cpu().detach().numpy()
    logging.info(f"Weight statistics:")
    logging.info(f"Mean weight: {np.mean(weights):.4f}")
    logging.info(f"Std weight: {np.std(weights):.4f}")
    logging.info(f"Min weight: {np.min(weights):.4f}")
    logging.info(f"Max weight: {np.max(weights):.4f}")
    
    # Analyze gate parameters
    gate_weights = model.gate_linear.weight.cpu().detach().numpy()
    logging.info(f"Gate weight statistics:")
    logging.info(f"Mean gate weight: {np.mean(gate_weights):.4f}")
    logging.info(f"Std gate weight: {np.std(gate_weights):.4f}")
    
    return {
        'weights': weights,
        'gate_weights': gate_weights
    }
