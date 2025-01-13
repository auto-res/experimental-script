import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim, seq_len):
        """Initialize the Learnable Gated Pooling module.
        
        Args:
            input_dim (int): Dimension of input vectors
            seq_len (int): Length of input sequence
        """
        super(LearnableGatedPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)  # Linear layer for gating
        
    def forward(self, x):
        """Forward pass of the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Pooled output tensor of shape (batch_size, input_dim)
        """
        # Apply learnable weights (element-wise multiplication, broadcasting across sequence length)
        weighted_x = x * self.weights
        
        # Calculate gate values using sigmoid activation
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)  # (batch_size, seq_len)
        
        # Apply gating mechanism (element-wise multiplication)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        
        # Perform average pooling operation
        pooled_vector = torch.mean(gated_x, dim=1)  # Average pooling across sequence length
        
        return pooled_vector

def test_learnable_gated_pooling():
    """Test function to verify the LearnableGatedPooling implementation."""
    # Example parameters
    input_dim = 768  # Example: BERT embedding dimension
    batch_size = 32
    seq_len = 10
    
    # Create random input embeddings
    embeddings = torch.randn(batch_size, seq_len, input_dim)
    
    # Initialize the model
    model = LearnableGatedPooling(input_dim, seq_len)
    
    # Forward pass
    pooled_output = model(embeddings)
    
    # Verify output shape
    expected_shape = (batch_size, input_dim)
    assert pooled_output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {pooled_output.shape}"
    
    print("LearnableGatedPooling test passed successfully!")
    return pooled_output

if __name__ == "__main__":
    test_learnable_gated_pooling()
