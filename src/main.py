import torch
from src.preprocess import generate_dummy_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Set parameters
    batch_size = 2
    seq_len = 10
    input_dim = 8
    
    # Generate data and run model
    data = generate_dummy_data(batch_size, seq_len, input_dim)
    output = train_model(data, input_dim, seq_len)
    evaluate_model(output)

if __name__ == "__main__":
    main()
