import os
import joblib

def load_model():
    model_path = r"C:\Users\marco\PycharmProjects\IMPRINT-app\models\imprint_extended.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    bundle = joblib.load(model_path)
    if not all(k in bundle for k in ["model","predictors","transforms"]):
        raise KeyError("Bundle must contain 'model', 'predictors', and 'transforms'.")
    return bundle["model"], bundle["predictors"], bundle["transforms"]