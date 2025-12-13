import os
import joblib

def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "imprint_extended.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["predictors"], bundle["transforms"]
