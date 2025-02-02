import torch
import numpy as np
from typing import Tuple

def generate_synthetic_data(
    num_samples: int,
    seq_len: int,
    input_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randn(num_samples, input_dim)
    return x, y
