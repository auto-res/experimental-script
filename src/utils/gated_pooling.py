import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGatedPooling(nn.Module):
    """
    Learnable Gated Pooling implementation.
    
    This module combines learnable weights with a gating mechanism to dynamically
    control the contribution of each element in the input sequence to the final
    pooled vector.
    
    Args:
        input_dim (int): Dimension of input vectors
        seq_len (int): Length of input sequence
    
    Mathematical representation:
    - Let x = [x_1, x_2, ..., x_n] be the input sequence
    - Let w = [w_1, w_2, ..., w_d] be the learnable weights
    - Let g = [g_1, g_2, ..., g_n] be the learnable gates
    
    The pooling operation is defined as:
    1. weighted_x = x * w  (element-wise multiplication)
    2. gated_x = weighted_x * sigmoid(g)
    3. pooled_vector = pooling_operation(gated_x)
    """
    def __init__(self, input_dim: int, seq_len: int):
        super(LearnableGatedPooling, self).__init__()
        # Initialize learnable weights with ones
        self.weights = nn.Parameter(torch.ones(input_dim))
        # Linear layer for computing gate values
        self.gate_linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Learnable Gated Pooling module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: Pooled output tensor of shape (batch_size, input_dim)
        """
        # Apply learnable weights (broadcasting across sequence length)
        weighted_x = x * self.weights
        
        # Compute gate values using linear layer and sigmoid activation
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)  # (batch_size, seq_len)
        
        # Apply gates to weighted input
        gated_x = weighted_x * gate_values.unsqueeze(2)
        
        # Average pooling across sequence length
        pooled_vector = torch.mean(gated_x, dim=1)
        
        return pooled_vector
