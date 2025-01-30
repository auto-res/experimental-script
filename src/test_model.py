import torch
from train import LearnableGatedPooling

def test_model():
    input_dim = 768  # Example: BERT embedding dimension
    batch_size = 32
    seq_len = 10
    embeddings = torch.randn(batch_size, seq_len, input_dim)

    learnable_gated_pooling = LearnableGatedPooling(input_dim, seq_len)
    pooled_output = learnable_gated_pooling(embeddings)

    print(f'Input shape: {embeddings.shape}')
    print(f'Output shape: {pooled_output.shape}')
    assert pooled_output.shape == (batch_size, input_dim), "Output shape mismatch"
    print('Test passed successfully!')

if __name__ == "__main__":
    test_model()
