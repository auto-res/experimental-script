from dataclasses import dataclass, field
from typing import List

@dataclass
class OptimizerConfig:
    lr: float = 5e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.5])
    weight_decay: float = 0.0
    eps: float = 1e-6

@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 10
    log_interval: int = 100
