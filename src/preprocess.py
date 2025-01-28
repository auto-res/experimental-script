import torch

def prepare_data(sequences, batch_size=32):
    dataset = torch.utils.data.TensorDataset(sequences)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
