
from fastapi import FastAPI, HTTPException, Request
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Load trained model
with open("property_valuation_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI(title="Real Estate AI - Property Valuation")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Define request model
class PropertyFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    property_age: int
    distance_to_city: float
    crime_rate: float
    school_quality: float

# Prediction endpoint
@app.post("/predict")
def predict_valuation(features: PropertyFeatures):
    try:
        input_data = np.array([[
            features.area, features.bedrooms, features.bathrooms,
            features.property_age, features.distance_to_city,
            features.crime_rate, features.school_quality
        ]])

        predicted_price = model.predict(input_data)[0]

        return {
            "predicted_valuation_AUD": round(predicted_price, 2),
            "input_features": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Home Page for UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides a PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
