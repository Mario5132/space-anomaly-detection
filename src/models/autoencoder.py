import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_loader import load_cmaps_data
from preprocessing import (
    add_rul_column, get_constant_sensors, remove_constant_sensors,
    create_anomaly_labels, prepare_features
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def build_autoencoder(input_dim):
    """
    Build a simple autoencoder for anomaly detection.
    """
    model = Sequential([
        # Encoder
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        # Decoder
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X_train, X_val, epochs=50, batch_size=32):
    """
    Train autoencoder on normal data only.
    """
    model = build_autoencoder(X_train.shape[1])
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def get_anomaly_scores(model, X):
    """
    Get reconstruction error for each sample.
    Higher error = more likely anomaly.
    """
    reconstructions = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    return mse

def find_threshold(scores, y_true):
    """
    Find optimal threshold using ROC curve.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold

def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_anomaly_scores(scores, y_true, threshold, save_path=None):
    """
    Plot anomaly scores distribution.
    """
    plt.figure(figsize=(10, 5))
    
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.4f}')
    
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("AUTOENCODER FOR ANOMALY DETECTION")
    print("="*60)
    
    # Load and preprocess
    print("\n1. Loading data...")
    train_df, test_df, rul_test = load_cmaps_data()
    
    print("\n2. Adding RUL column...")
    train_df = add_rul_column(train_df)
    
    print("\n3. Removing constant sensors...")
    constant_sensors = get_constant_sensors(train_df)
    print(f"   Removing: {constant_sensors}")
    train_df = remove_constant_sensors(train_df, constant_sensors)
    
    print("\n4. Creating anomaly labels...")
    train_df = create_anomaly_labels(train_df, rul_threshold=30)
    
    # Prepare features
    print("\n5. Preparing features...")
    feature_cols = [col for col in train_df.columns if col not in 
                    ['unit_number', 'time_in_cycles', 'RUL', 'is_anomaly']]
    X = train_df[feature_cols].values
    y = train_df['is_anomaly'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Anomaly percentage: {y.mean()*100:.2f}%")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train autoencoder on NORMAL data only
    print("\n7. Training autoencoder on NORMAL data only...")
    X_train_normal = X_train_scaled[y_train == 0]
    X_val_normal = X_val_scaled[y_val == 0]
    
    print(f"   Training on {X_train_normal.shape[0]} normal samples")
    print(f"   Validating on {X_val_normal.shape[0]} normal samples")
    
    model, history = train_autoencoder(X_train_normal, X_val_normal)
    
    # Get anomaly scores
    print("\n8. Computing anomaly scores...")
    train_scores = get_anomaly_scores(model, X_train_scaled)
    val_scores = get_anomaly_scores(model, X_val_scaled)
    
    # Find threshold
    threshold = find_threshold(val_scores, y_val)
    print(f"   Optimal threshold: {threshold:.6f}")
    
    # Predict on validation
    y_pred = (val_scores > threshold).astype(int)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"       Anomaly   {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_val, val_scores):.4f}")
    
    # Save figures
    os.makedirs('results/figures', exist_ok=True)
    plot_training_history(history, save_path='results/figures/autoencoder_training.png')
    plot_anomaly_scores(val_scores, y_val, threshold, save_path='results/figures/anomaly_scores.png')
    
    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - Autoencoder')
    plt.savefig('results/figures/autoencoder_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("AUTOENCODER TRAINING COMPLETE")
    print("="*60)