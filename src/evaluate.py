import torch
from tqdm import tqdm

def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, torch.mean(inputs, dim=1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return {'avg_loss': avg_loss}
