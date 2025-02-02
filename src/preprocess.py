import torch
import numpy as np
from typing import Tuple
import yaml
from pathlib import Path

def load_config() -> dict:
    """Load configuration from yaml file."""
    config_path = Path("config/model_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def generate_sample_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for testing the model."""
    config = load_config()
    size = config['data']['train_size']
    input_dim = config['model']['input_dim']
    seq_len = config['model']['seq_len']
    
    features = torch.randn(size, seq_len, input_dim)
    targets = torch.randint(0, 2, (size,))
    
    return features, targets
