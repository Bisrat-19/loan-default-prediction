from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_model(y_true, y_pred):
    """Evaluate the model using multiple metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics."""
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")