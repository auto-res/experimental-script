from pathlib import Path
from train import train_model
from evaluate import evaluate_model

def main():
    config_path = Path('config') / 'model_config.yaml'
    
    print("Starting training...")
    train_model(config_path)
    
    print("\nStarting evaluation...")
    evaluate_model(config_path)

if __name__ == "__main__":
    main()
