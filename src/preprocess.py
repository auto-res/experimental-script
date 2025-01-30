import numpy as np

def load_and_preprocess_data():
    """Generate sample data for demonstration of Learnable Gated Pooling."""
    data = np.random.randn(32, 10, 768).astype('float32')
    return data
