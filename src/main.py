import torch
from pathlib import Path
import json
from train import train_model
from evaluate import evaluate_model

def main():
    """Main function to run the experiment."""
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Train model
    print("Training model...")
    model, train_metrics = train_model()
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_metrics = evaluate_model(model)
    
    # Save metrics
    all_metrics = {
        'training': train_metrics,
        'evaluation': eval_metrics
    }
    with open('logs/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), 'models/learnable_gated_pooling.pt')
    print("\nResults:")
    print(f"Final training loss: {train_metrics['loss'][-1]:.4f}")
    print(f"Evaluation accuracy: {eval_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
