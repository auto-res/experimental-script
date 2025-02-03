import torch
import torch.nn as nn
import torch.nn.functional as F
from config.default import OptimizerConfig, TrainingConfig
from .preprocess import get_mnist_data
from .train import train_epoch
from .evaluate import evaluate
from .optimizers.aggmo_madgrad import AggMoMADGRAD

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    opt_config = OptimizerConfig()
    train_config = TrainingConfig()
    
    # Get data
    train_loader, test_loader = get_mnist_data(train_config.batch_size)
    
    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = AggMoMADGRAD(
        model.parameters(),
        lr=opt_config.lr,
        betas=opt_config.betas,
        weight_decay=opt_config.weight_decay,
        eps=opt_config.eps
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(train_config.epochs):
        print(f'\nEpoch: {epoch}')
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, train_config.log_interval
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        print(f'Training Loss: {train_metrics["loss"]:.6f}, '
              f'Training Accuracy: {train_metrics["accuracy"]:.2f}%')
        print(f'Test Loss: {test_metrics["loss"]:.6f}, '
              f'Test Accuracy: {test_metrics["accuracy"]:.2f}%')

if __name__ == '__main__':
    main()
