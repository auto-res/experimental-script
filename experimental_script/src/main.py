from .preprocess import load_and_preprocess_data
from .train import LearnableGatedPooling, train_model
from .evaluate import evaluate_model
from ..config.model_config import ModelConfig

def main():
    x_train, y_train = load_and_preprocess_data()
    x_test, y_test = load_and_preprocess_data()
    
    model = LearnableGatedPooling(ModelConfig.input_dim)
    model, losses = train_model(model, x_train, y_train)
    
    metrics = evaluate_model(model, x_test, y_test)
    print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()
