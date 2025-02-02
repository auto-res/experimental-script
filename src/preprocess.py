from typing import Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader
