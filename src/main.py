from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("Starting experiment...")
    model = train_model()
    evaluate_model(model)
    print("Experiment completed!")
