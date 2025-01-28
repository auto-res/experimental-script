import torch
import numpy as np

def load_data():
    train_data = torch.randn(100, 10, 768)  # 100 samples, 10 seq_len, 768 features
    train_labels = torch.randint(0, 2, (100,))  # binary classification
    return train_data, train_labels
