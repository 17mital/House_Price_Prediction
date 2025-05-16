import joblib
import pandas as pd

# Load the trained scaler
scaler = joblib.load("scaler.pkl")

# Retrieve feature names used during training
model_features = scaler.feature_names_in_
print("Features used in training:", model_features)
