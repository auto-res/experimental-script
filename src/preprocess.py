import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data['input'])
    
    def __getitem__(self, idx):
        return {
            'input': self.data['input'][idx],
            'target': self.data['target'][idx]
        }

def create_dataloaders(config):
    train_dataset = SequenceDataset(config['data']['train_path'])
    val_dataset = SequenceDataset(config['data']['val_path'])
    test_dataset = SequenceDataset(config['data']['test_path'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    return train_loader, val_loader, test_loader
