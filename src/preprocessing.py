import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the loan dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handle missing values and clean the dataset."""
    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def encode_categorical(df):
    """Encode categorical variables."""
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(X):
    """Scale numerical features."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def preprocess_data(file_path, target_column='default'):
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables
    X = encode_categorical(X)
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler