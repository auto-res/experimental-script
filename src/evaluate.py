import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            pred = (torch.sigmoid(output.squeeze()) > 0.5).int()
            predictions.extend(pred.tolist())
            targets.extend(batch_y.tolist())
    
    accuracy = accuracy_score(targets, predictions)
    return accuracy
