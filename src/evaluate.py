import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            predictions.extend(output.numpy())
            targets.extend(batch_y.numpy())
            
    mse = mean_squared_error(targets, predictions)
    return {'mse': mse, 'predictions': predictions, 'targets': targets}
