# Scripts for evaluation.
import torch
from pathlib import Path
import yaml
from models.learnable_gated_pooling import LearnableGatedPooling
from preprocess import load_data
from sklearn.metrics import accuracy_score

def evaluate_model(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = LearnableGatedPooling(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len']
    )
    
    model.load_state_dict(torch.load(Path('models') / 'gated_pooling.pth'))
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    features, labels = load_data(Path('data'))
    features = features.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy()
        
    accuracy = accuracy_score(labels.numpy(), predictions)
    print(f'Model Accuracy: {accuracy:.4f}')
