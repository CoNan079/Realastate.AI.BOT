import pickle
import numpy as np
import sklearn
from sklearn.utils import _IS_32BIT
from sklearn.linear_model import LinearRegression

# Ensure compatibility when loading the pickle file
def load_model(filename):
    with open(filename, "rb") as f:
        try:
            model = pickle.load(f)
        except AttributeError:
            import sklearn.ensemble._gradient_boosting
            model = pickle.load(f)
        return model

# Load the model safely
model = load_model("model.pkl")

# Example API using FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Real Estate AI Bot is running"}

@app.post("/predict")
def predict(features: list):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}