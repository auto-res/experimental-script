from typing import Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    train_size = int(len(dataset) * config['data']['train_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    return train_loader, val_loader
