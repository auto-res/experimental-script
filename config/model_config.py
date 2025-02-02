from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_dim: int = 768
    seq_len: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
