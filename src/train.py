# Scripts for training models.
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from tqdm import tqdm
from models.learnable_gated_pooling import LearnableGatedPooling
from preprocess import load_data

def train_model(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = LearnableGatedPooling(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len']
    )
    
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    features, labels = load_data(Path('data'))
    features = features.to(device)
    labels = labels.to(device)
    
    for epoch in tqdm(range(config['training']['num_epochs'])):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.float())
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), Path('models') / 'gated_pooling.pth')
