# House Prices Prediction web site

This project implements a machine learning pipeline to predict house prices using the California Housing dataset. It includes data processing, model training, API deployment, and a web interface, leveraging MLOps tools.

## Features
- Uses all 8 features of the California Housing dataset (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
- Trains a `RandomForestRegressor` model.
- Deployable web app with Streamlit for user input and predictions.
- REST API with FastAPI for model inference.

## Requirements
- Python 3.10
- Docker and Docker Compose
- Packages listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd house_prices_prediction
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

1. Train the model:
    ```bash
    python code/models/train.py
    ```
2. Start the application and API:
    ```bash
    cd code/deployment
    docker-compose up --build
    ```

3. Access the web app at http://localhost:8501.
4. View API docs at http://localhost:8000/docs.
