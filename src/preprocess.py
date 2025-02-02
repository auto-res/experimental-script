# Scripts for data preprocessing.
import torch
import numpy as np
from typing import Tuple
from pathlib import Path

def load_data(data_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess data for training.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (features, labels)
    """
    features = torch.randn(1000, 10, 768)  # 1000 samples, seq_len=10, dim=768
    labels = torch.randint(0, 2, (1000,))  # Binary classification for testing
    return features, labels
