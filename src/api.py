from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# File paths
MODEL_PATH = "models/housing_model.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# Define input data schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Health check
@app.get("/")
def read_root():
    return {"message": "Housing Price Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])
    # Make prediction
    prediction = model.predict(data)[0]
    return {"predicted_price": prediction}
