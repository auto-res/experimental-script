import torch
from preprocess import load_data
from train import LearnableGatedPooling

def evaluate_model(model=None):
    data, labels = load_data()
    
    if model is None:
        model = LearnableGatedPooling(768, 10)
    
    model.eval()
    with torch.no_grad():
        outputs = model(data).squeeze()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        print(f"Evaluation accuracy: {accuracy.item():.4f}")
    
    return accuracy.item()
