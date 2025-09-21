import streamlit as st
import requests

# Page Settings
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("House Price Predictor")
st.markdown("Enter realistic housing features to predict the median house value in dollars.")

# Center container for input
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.number_input("Median Income (in $1,000)", min_value=5.0, max_value=150.0, value=30.0, step=0.1, format="%.1f")
        HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100, value=20, step=1, format="%d")
        AveRooms = st.number_input("Average Rooms", min_value=1, max_value=10, value=5, step=1, format="%d")
        AveBedrms = st.number_input("Average Bedrooms", min_value=1, max_value=6, value=2, step=1, format="%d")
    with col2:
        Population = st.number_input("Population (hundreds)", min_value=10, max_value=5000, value=100, step=10, format="%d")
        AveOccup = st.number_input("Average Occupancy", min_value=1, max_value=10, value=3, step=1, format="%d")
        Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1, format="%.1f")
        Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1, format="%.1f")

# Prediction button
if st.button("Predict House Value"):
    data = {
        "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms, "AveBedrms": AveBedrms,
        "Population": Population, "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
    }
    try:
        response = requests.post("http://api:8000/predict", json=data)
        response.raise_for_status()
        prediction = response.json()["prediction"] * 100000
        st.success(f"Predicted House Value: ${prediction:,.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {str(e)}")

# Style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .element-container {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Extra information
st.markdown("**Notes:** Values are based on the California Housing dataset. Predictions are in USD.")