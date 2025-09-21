import streamlit as st
import requests
import json

st.title("House Price Predictor")

MedInc = st.number_input("Median Income", value=3.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict"):
    data = {
        "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms, "AveBedrms": AveBedrms,
        "Population": Population, "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
    }
    response = requests.post("http://api:8000/predict", json=data)
    prediction = response.json()["prediction"]
    st.success(f"Predicted House Value (in $100,000s): {prediction:.2f}")