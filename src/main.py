import torch
import torch.nn as nn
import yaml
from train import LearnableGatedPooling, train_model
from evaluate import evaluate_model
from preprocess import create_dataloaders

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = LearnableGatedPooling(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len']
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    train_model(model, train_loader, val_loader, criterion, optimizer, config)
    
    metrics = evaluate_model(model, test_loader, config)
    print('Test metrics:', metrics)

if __name__ == '__main__':
    main()
