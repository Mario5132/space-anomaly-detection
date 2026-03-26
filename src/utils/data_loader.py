import pandas as pd
import numpy as np
import os

def load_cmaps_data(data_path='data/raw'):
    """
    Load NASA CMAPSS turbofan dataset.
    Returns train_data, test_data, rul_test
    """
    
    # Column names for all 26 columns
    column_names = [
        'unit_number', 'time_in_cycles',
        'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
    ]
    
    # Load training data with proper delimiter (fixed width, space separated)
    train_path = os.path.join(data_path, 'train_FD001.txt')
    train_df = pd.read_csv(train_path, sep='\s+', header=None, names=column_names)
    
    # Load test data
    test_path = os.path.join(data_path, 'test_FD001.txt')
    test_df = pd.read_csv(test_path, sep='\s+', header=None, names=column_names)
    
    # Load true RUL values
    rul_path = os.path.join(data_path, 'RUL_FD001.txt')
    rul_test = pd.read_csv(rul_path, sep='\s+', header=None, names=['rul_true'])
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"RUL test shape: {rul_test.shape}")
    
    # Convert all columns to numeric (they might be object type)
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    for col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Drop any columns that are all NaN
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')
    
    return train_df, test_df, rul_test

if __name__ == "__main__":
    train, test, rul = load_cmaps_data()
    print("\nFirst 5 rows of training data:")
    print(train.head())
    print("\nData types:")
    print(train.dtypes.head())
    print(f"\nUnit numbers: {train['unit_number'].unique()[:10]}")
    print(f"Time cycles range: {train['time_in_cycles'].min()} to {train['time_in_cycles'].max()}")