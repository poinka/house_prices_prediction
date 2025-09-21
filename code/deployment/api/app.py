from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

# Load model with absolute path
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models'))
model_path = os.path.join(models_dir, 'model.pkl')
model = joblib.load(model_path)

# Define input schema (all 8 features)
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HousingFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
                      features.Population, features.AveOccup, features.Latitude, features.Longitude]])
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}