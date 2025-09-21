import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import ssl
import certifi
import urllib.request
import os

# Create a custom HTTPS handler with certifi's CA bundle
context = ssl.create_default_context(cafile=certifi.where())
https_handler = urllib.request.HTTPSHandler(context=context)
opener = urllib.request.build_opener(https_handler)
urllib.request.install_opener(opener)

# Load data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train (using RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse}")

# Save model with a corrected path
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")