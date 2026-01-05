import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# File paths
RAW_DATA_PATH = "data/processed/housing_processed.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "housing_model.pkl")
METRICS_PATH = "metrics/train_metrics.json"

# Create folders if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

# Load processed data
data = pd.read_csv(RAW_DATA_PATH)

# Split features and target
X = data.drop(columns=["MedHouseVal"])
y = data["MedHouseVal"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# Save trained model
joblib.dump(model, MODEL_PATH)

print("Model trained and saved!")
print("Metrics:", metrics)
