import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_mnist_loaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        'data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
