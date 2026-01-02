# src/data_preprocessing.py
import pandas as pd
import os

# Load raw data
raw_data_path = os.path.join("data", "raw", "housing_raw.csv")
df = pd.read_csv(raw_data_path)

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Example preprocessing (for now, just save the cleaned CSV)
processed_data_path = os.path.join("data", "processed", "housing_processed.csv")
df.to_csv(processed_data_path, index=False)

print(f"Processed data saved to {processed_data_path}")
