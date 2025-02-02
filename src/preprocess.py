import torch
import numpy as np
from typing import Tuple

def generate_synthetic_data(
    batch_size: int,
    seq_len: int,
    input_dim: int
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, input_dim)

def prepare_data(
    data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]
