import torch
from src.preprocess import load_data
from src.train import LearnableGatedPooling

def evaluate_model():
    X, y = load_data()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    model = LearnableGatedPooling(input_dim=768)
    outputs = model(X_tensor)
    logits = model.classifier(outputs)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y_tensor).float().mean().item()
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy
