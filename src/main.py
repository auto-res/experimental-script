import torch
import matplotlib.pyplot as plt
from config.train_config import TrainConfig
from src.preprocess import get_mnist_loaders
from src.train import train

def plot_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    for name, metrics in results.items():
        ax1.plot(metrics['train_loss'], label=name)
        ax2.plot(metrics['train_acc'], label=name)
        ax3.plot(metrics['test_loss'], label=name)
        ax4.plot(metrics['test_acc'], label=name)
    
    ax1.set_title('Training Loss')
    ax2.set_title('Training Accuracy')
    ax3.set_title('Test Loss')
    ax4.set_title('Test Accuracy')
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results.png')

def main():
    config = TrainConfig()
    results = train(config)
    plot_results(results)
    print("\nExperiment completed! Results saved to results.png")

if __name__ == "__main__":
    main()
