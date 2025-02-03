import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Any
from .models.simple_cnn import SimpleCNN
from .optimizers.modag import MoDAG
from .evaluate import evaluate
from .preprocess import get_mnist_loaders

def train_epoch(model: nn.Module, 
                optimizer: torch.optim.Optimizer,
                train_loader: torch.utils.data.DataLoader,
                device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1),
                            'acc': 100. * correct / total})
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }

def train(config: Any) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(config.batch_size, config.num_workers)
    model = SimpleCNN(config.hidden_size, config.num_classes).to(device)
    
    optimizers = {
        'MoDAG': MoDAG(model.parameters(), lr=config.learning_rate,
                       beta_ag=config.beta_ag, adap_factor=config.adap_factor),
        'Adam': torch.optim.Adam(model.parameters(), lr=config.learning_rate),
        'SGD': torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    }
    
    results = {name: {'train_loss': [], 'train_acc': [], 
                     'test_loss': [], 'test_acc': []}
              for name in optimizers}
    
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name} optimizer")
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        for epoch in range(config.epochs):
            metrics = train_epoch(model, optimizer, train_loader, device)
            results[name]['train_loss'].append(metrics['loss'])
            results[name]['train_acc'].append(metrics['accuracy'])
            
            test_metrics = evaluate(model, test_loader, device)
            results[name]['test_loss'].append(test_metrics['loss'])
            results[name]['test_acc'].append(test_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}/{config.epochs}")
            print(f"Train Loss: {metrics['loss']:.4f}, Train Acc: {metrics['accuracy']:.2f}%")
            print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.2f}%")
    
    return results
