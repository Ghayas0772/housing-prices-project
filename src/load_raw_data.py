# src/load_raw_data.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

# Load California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Define path to save raw data
raw_data_path = os.path.join("data", "raw", "housing_raw.csv")

# Save as CSV
df.to_csv(raw_data_path, index=False)

print(f"Raw data saved to {raw_data_path}")


import pandas as pd

# Load the raw CSV
df = pd.read_csv("data/raw/housing_raw.csv")

# Show first 5 rows
print(df.head())
