import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save the trained model."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load a trained model."""
    return joblib.load(file_path)

def predict(model, X):
    """Make predictions with the trained model."""
    return model.predict(X)