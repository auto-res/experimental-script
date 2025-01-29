import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_dataloaders(sequences, labels, batch_size=32):
    dataset = SequenceDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
