import torch

def prepare_data(sequences, batch_size=32):
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=batch_size, shuffle=True)
    return dataloader
