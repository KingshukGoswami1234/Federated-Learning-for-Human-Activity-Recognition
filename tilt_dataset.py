import pandas as pd
import numpy as np
import os

def add_tilt_features_vectorized(df):
    """
    Calculates pitch and roll from the accelerometer axes.
    Appends columns in the specific order: pitch_1...pitch_60, then roll_1...roll_60.
    """
    pitch_cols = {}
    roll_cols = {}
    
    # Iterate 1 to 60 to match the fl_train.csv column naming convention
    for i in range(1, 61):
        ax = df[f'acc_x_{i}']
        ay = df[f'acc_y_{i}']
        az = df[f'acc_z_{i}']
        
        # Calculate pitch and roll in degrees[cite: 2]
        pitch_cols[f'pitch_{i}'] = np.arctan2(ax, np.sqrt(ay**2 + az**2)) * (180 / np.pi)
        roll_cols[f'roll_{i}'] = np.arctan2(ay, np.sqrt(ax**2 + az**2)) * (180 / np.pi)
        
    # Merge dictionaries sequentially to enforce the pitch_1..60, roll_1..60 ordering
    new_cols = {**pitch_cols, **roll_cols}
    tilt_df = pd.DataFrame(new_cols)
    
    return pd.concat([df, tilt_df], axis=1)

print("Pre-computing tilt features for all splits...")

# Target the newly generated federated dataset splits
file_paths = [
    "fl_train.csv",
    "fl_validation.csv",
    "fl_test.csv"
]

for file_path in file_paths:
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)
        df = add_tilt_features_vectorized(df)
        
        # Extract the directory and filename separately to create a valid save path[cite: 2]
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Handle cases where files are in the current working directory
        if directory == "":
            new_filepath = f"tilt_{filename}"
        else:
            new_filepath = os.path.join(directory, f"tilt_{filename}")
        
        df.to_csv(new_filepath, index=False)
        print(f"Saved successfully to: {new_filepath}")
    else:
        print(f"File not found (skipped): {file_path}")

print("Tilt datasets generated successfully.")