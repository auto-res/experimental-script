import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple

from models.learnable_gated_pooling import LearnableGatedPooling
from preprocess import load_config, generate_sample_data

def train_model() -> Tuple[LearnableGatedPooling, dict]:
    """Train the LearnableGatedPooling model."""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = LearnableGatedPooling(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len']
    ).to(device)
    
    # Generate data
    features, targets = generate_sample_data()
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    metrics = {'loss': []}
    for epoch in range(config['training']['num_epochs']):
        epoch_loss = 0.0
        for batch_features, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_features.to(device))
            loss = criterion(output, batch_targets.float().to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        metrics['loss'].append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Loss: {metrics['loss'][-1]:.4f}")
    
    return model, metrics
