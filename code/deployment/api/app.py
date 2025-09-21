from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
import pandas as pd

app = FastAPI()

# Load model with absolute path
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models'))
model_path = os.path.join(models_dir, 'model.pkl')
model = joblib.load(model_path)

# Define feature names (all 8 features)
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

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
    # Ensure feature names are preserved (create a DataFrame)
    data_df = pd.DataFrame(data, columns=feature_names)
    prediction = model.predict(data_df)[0]
    return {"prediction": float(prediction)}