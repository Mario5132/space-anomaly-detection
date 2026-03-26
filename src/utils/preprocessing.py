import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_rul_column(df):
    """
    Add Remaining Useful Life (RUL) column.
    For each engine, RUL = max_cycles - current_cycle
    """
    df_with_rul = df.copy()
    max_cycles = df_with_rul.groupby('unit_number')['time_in_cycles'].transform('max')
    df_with_rul['RUL'] = max_cycles - df_with_rul['time_in_cycles']
    return df_with_rul

def get_constant_sensors(df):
    """
    Identify sensors with constant values (no variation).
    These provide no information and should be removed.
    """
    constant_sensors = []
    for col in df.columns:
        if col.startswith('sensor_'):
            if df[col].nunique() == 1:
                constant_sensors.append(col)
    return constant_sensors

def remove_constant_sensors(df, constant_sensors):
    """Remove constant sensors from dataframe."""
    return df.drop(columns=constant_sensors)

def create_anomaly_labels(df, rul_threshold=30):
    """
    Create labels for anomaly detection.
    1 = anomaly (RUL <= threshold, engine close to failure)
    0 = normal (RUL > threshold)
    """
    df_with_labels = df.copy()
    df_with_labels['is_anomaly'] = (df_with_labels['RUL'] <= rul_threshold).astype(int)
    return df_with_labels

def prepare_features(df, exclude_cols=['unit_number', 'time_in_cycles', 'RUL', 'is_anomaly']):
    """
    Prepare feature matrix by excluding identifier columns.
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return df[feature_cols]

def scale_features(train_df, test_df, feature_cols):
    """
    Scale features using StandardScaler fitted on training data.
    """
    scaler = StandardScaler()
    
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    return train_scaled, test_scaled, scaler

if __name__ == "__main__":
    # Quick test
    from data_loader import load_cmaps_data
    
    train_df, test_df, rul_test = load_cmaps_data()
    
    print("\n--- Adding RUL column ---")
    train_df = add_rul_column(train_df)
    print(f"Train shape after adding RUL: {train_df.shape}")
    print(train_df[['unit_number', 'time_in_cycles', 'RUL']].head())
    
    print("\n--- Identifying constant sensors ---")
    constant_sensors = get_constant_sensors(train_df)
    print(f"Constant sensors: {constant_sensors}")
    
    if constant_sensors:
        train_df = remove_constant_sensors(train_df, constant_sensors)
        print(f"Train shape after removing constant sensors: {train_df.shape}")
    
    print("\n--- Creating anomaly labels (RUL <= 30 = anomaly) ---")
    train_df = create_anomaly_labels(train_df, rul_threshold=30)
    print(f"Anomaly distribution:")
    print(train_df['is_anomaly'].value_counts())