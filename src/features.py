import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw/turbofan/"
PROCESSED_DIR = "data/processed"
COLUMNS = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's{i}' for i in range(1, 22)]
USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's10', 's16', 's18', 's19']
WINDOWS = [5, 10, 30]

def load_and_clean_data(file_name):
    """Loads raw txt, assigns columns, and drops zero-variance sensors."""
    path = os.path.join(RAW_DIR, file_name)
    df = pd.read_csv(path, sep='\s+', header=None, names=COLUMNS)
    df.drop(columns=DROP_SENSORS, inplace=True)
    return df

def generate_rolling_features(df):
    """Calculates rolling mean, std, and diff per engine efficiently."""
    df = df.sort_values(['engine_id', 'cycle']).copy()
    
    # Dictionary to hold our new features before attaching them
    new_features = {}
    
    for w in WINDOWS:
        for s in USEFUL_SENSORS:
            # Rolling Mean
            new_features[f'{s}_mean_{w}'] = df.groupby('engine_id')[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            
            # Rolling Standard Deviation
            new_features[f'{s}_std_{w}'] = df.groupby('engine_id')[s].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            
            # Rolling Difference (Rate of Change) - Fixed deprecation warning here
            new_features[f'{s}_diff_{w}'] = df.groupby('engine_id')[s].transform(
                lambda x: x - x.shift(w).bfill()
            )
            
    # Convert the dictionary of new columns into a dataframe
    new_features_df = pd.DataFrame(new_features, index=df.index)
    
    # Concatenate everything at once to prevent memory fragmentation
    df = pd.concat([df, new_features_df], axis=1)
            
    return df

def process_training_data():
    print("Processing Training data")
    train_df = load_and_clean_data('train_FD001.txt')

    # Target Computation (RUL)
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    train_df = train_df.merge(max_cycles, on='engine_id')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

    # Clip RUL at 125
    train_df['RUL'] = train_df['RUL'].clip(upper=125)

    # Cycle Normalisation (0 = start, 1 = failure)
    train_df['cycle_norm'] = train_df['cycle'] / train_df['max_cycle']
    train_df.drop(columns=['max_cycle'], inplace=True)

    # Generate Features
    train_df = generate_rolling_features(train_df)

    # Save to Parquet
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_parquet(os.path.join(PROCESSED_DIR, 'features_train.parquet'), index=False)
    print(f"Saved features_train.parquet | Shape: {train_df.shape}")


def process_test_data():
    print("Processing Test Data...")
    test_df = load_and_clean_data('test_FD001.txt')
    
    # Load true RUL values for the *last* cycle of each test engine
    true_rul = pd.read_csv(os.path.join(RAW_DIR, 'RUL_FD001.txt'), sep='\s+', header=None, names=['true_RUL'])
    true_rul['engine_id'] = true_rul.index + 1
    
    # To calculate RUL for every row in the test set, we find the max observed cycle 
    # and add the true remaining RUL to get the absolute max cycle of that engine.
    max_test_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    max_test_cycles = max_test_cycles.merge(true_rul, on='engine_id')
    max_test_cycles['absolute_max_cycle'] = max_test_cycles['cycle'] + max_test_cycles['true_RUL']
    
    test_df = test_df.merge(max_test_cycles[['engine_id', 'absolute_max_cycle']], on='engine_id')
    test_df['RUL'] = test_df['absolute_max_cycle'] - test_df['cycle']
    
    # Clip RUL at 125
    test_df['RUL'] = test_df['RUL'].clip(upper=125)
    test_df.drop(columns=['absolute_max_cycle'], inplace=True)
    
    # Generate Features (Note: cycle_norm is NOT computed for test data per requirements)
    test_df = generate_rolling_features(test_df)
    
    # Save to Parquet
    test_df.to_parquet(os.path.join(PROCESSED_DIR, 'features_test.parquet'), index=False)
    print(f"Saved features_test.parquet | Shape: {test_df.shape}")

if __name__ == "__main__":
    process_training_data()
    process_test_data()
    print("Feature engineering complete!")