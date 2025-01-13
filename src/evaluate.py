# Scripts for evaluation.
import torch
import torch.nn as nn
from models.learnable_gated_pooling import LearnableGatedPooling
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

__all__ = ['evaluate_model']

def evaluate_model(model, test_loader, device=None):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained LearnableGatedPooling model
        test_loader: DataLoader for test data
        device: torch device
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join('logs', 'evaluation.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Log results
    for metric_name, value in metrics.items():
        logging.info(f'{metric_name.capitalize()}: {value:.4f}')
    
    return metrics
