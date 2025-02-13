from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import os

app = FastAPI()

# Define the path to the trained model file
MODEL_PATH = "property_valuation_model.pkl"

# Ensure the model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure it's uploaded to the repository.")

# Load the trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

@app.get("/")
def home():
    """Health check endpoint"""
    return {"message": "Real Estate AI Bot is running successfully!"}

@app.get("/predict")
def predict_property_value(area: float, bedrooms: int):
    """
    Predict property value based on input features.
    
    Example request:
    GET /predict?area=200&bedrooms=3
    """
    try:
        # Ensure input is formatted correctly
        input_data = np.array([[area, bedrooms]])
        
        # Predict using the loaded model
        predicted_value = model.predict(input_data)[0]

        return {"predicted_value": round(predicted_value, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")