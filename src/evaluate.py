# Scripts for evaluation.
import torch

def evaluate_model(model_output):
    print(f"Output shape: {model_output.shape}")
    print(f"Output mean: {torch.mean(model_output):.4f}")
    print(f"Output std: {torch.std(model_output):.4f}")
    return model_output.shape
