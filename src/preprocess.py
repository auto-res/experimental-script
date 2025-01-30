import numpy as np

def load_data():
    X = np.random.randn(100, 10, 768)  # (num_samples, seq_len, input_dim)
    y = np.random.randint(0, 2, size=(100,))  # binary classification
    return X, y
