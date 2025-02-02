import torch

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        output = model(test_data)
        mse_loss = torch.mean((output - torch.mean(test_data, dim=1)) ** 2)
        print(f"Test MSE Loss: {mse_loss.item():.6f}")
    return mse_loss.item()
