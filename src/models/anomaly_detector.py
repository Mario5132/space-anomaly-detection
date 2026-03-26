import pandas as pd
import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_loader import load_cmaps_data
from preprocessing import (
    add_rul_column, get_constant_sensors, remove_constant_sensors,
    create_anomaly_labels, prepare_features, scale_features
)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(X_train, contamination=0.15, random_state=42):
    """
    Train Isolation Forest for anomaly detection.
    contamination = expected proportion of anomalies
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_true):
    """
    Evaluate model and print metrics.
    """
    # Predict (-1 for anomaly, 1 for normal)
    y_pred_binary = model.predict(X_test)
    # Convert to 0/1 (1 = anomaly, 0 = normal)
    y_pred = np.where(y_pred_binary == -1, 1, 0)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"       Anomaly   {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    return y_pred, cm

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - Isolation Forest')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("ANOMALY DETECTION FOR SPACECRAFT TELEMETRY")
    print("Using NASA CMAPSS Turbofan Dataset")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    train_df, test_df, rul_test = load_cmaps_data()
    
    # Preprocess
    print("\n2. Adding RUL column...")
    train_df = add_rul_column(train_df)
    
    print("\n3. Removing constant sensors...")
    constant_sensors = get_constant_sensors(train_df)
    print(f"   Removing: {constant_sensors}")
    train_df = remove_constant_sensors(train_df, constant_sensors)
    
    print("\n4. Creating anomaly labels (RUL <= 30 = anomaly)...")
    train_df = create_anomaly_labels(train_df, rul_threshold=30)
    
    # Prepare features
    print("\n5. Preparing features...")
    feature_cols = [col for col in train_df.columns if col not in 
                    ['unit_number', 'time_in_cycles', 'RUL', 'is_anomaly']]
    X = train_df[feature_cols].values
    y = train_df['is_anomaly'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Anomaly percentage: {y.mean()*100:.2f}%")
    
    # Split into train/validation (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    # Train model
    print("\n6. Training Isolation Forest...")
    model = train_isolation_forest(X_train, contamination=y_train.mean())
    
    # Evaluate
    print("\n7. Evaluating on validation set...")
    y_pred, cm = evaluate_model(model, X_val, y_val)
    
    # Save confusion matrix
    os.makedirs('results/figures', exist_ok=True)
    plot_confusion_matrix(cm, save_path='results/figures/confusion_matrix.png')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)