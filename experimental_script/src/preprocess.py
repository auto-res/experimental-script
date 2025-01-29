import torch
from typing import Tuple
from config.model_config import ModelConfig

def load_and_preprocess_data() -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(ModelConfig.batch_size, ModelConfig.seq_len, ModelConfig.input_dim)
    y = torch.randn(ModelConfig.batch_size, ModelConfig.input_dim)
    return x, y
