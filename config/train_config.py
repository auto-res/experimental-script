from dataclasses import dataclass
from typing import List
import torch

@dataclass
class TrainConfig:
    # Dataset
    batch_size: int = 128
    num_workers: int = 4
    
    # Optimizer
    learning_rate: float = 1e-3
    betas: List[float] = [0.0, 0.99]
    weight_decay: float = 0.0
    adaptive_rate: float = 1.0
    
    # Training
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100
