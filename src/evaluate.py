import torch

def evaluate_model(model, test_data):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_data:
            output = model(batch)
            target = torch.mean(batch, dim=1)
            loss = torch.nn.functional.mse_loss(output, target)
            total_loss += loss.item()
    
    return total_loss / len(test_data)
