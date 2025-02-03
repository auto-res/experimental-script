from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    # Dataset
    batch_size: int = 128
    num_workers: int = 2
    
    # Training
    epochs: int = 10
    learning_rate: float = 0.01
    beta_ag: List[float] = field(default_factory=lambda: [0.9, 0.99])
    weight_decay: float = 0.0
    adap_factor: float = 0.1
    
    # Model
    hidden_size: int = 64
    num_classes: int = 10
