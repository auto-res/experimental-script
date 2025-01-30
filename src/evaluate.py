import torch
from train import LearnableGatedPooling

def evaluate_model():
    """Evaluate the Learnable Gated Pooling model."""
    # Create test data
    test_data = torch.randn(16, 10, 768)
    
    # Initialize model
    model = LearnableGatedPooling(768, 10)
    
    # Get pooled output
    with torch.no_grad():
        output = model(test_data)
    
    # Print shapes for verification
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Gate weights shape: {model.weights.shape}")
    return output
