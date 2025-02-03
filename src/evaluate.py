import torch
from torch import nn
from typing import Dict
from tqdm import tqdm

def evaluate(model: nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device: torch.device) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                pbar.set_postfix({'loss': test_loss / (pbar.n + 1),
                                'acc': 100. * correct / total})
    
    return {
        'loss': test_loss / len(test_loader),
        'accuracy': 100. * correct / total
    }
