import joblib
import numpy as np
from stable_baselines3 import PPO
import pandas as pd

# Load the saved model, encoders, scaler, and feature columns
model = PPO.load("crop_rotation_model")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Define new input
input_data = {
    'Region': 'North',                 # Categorical
    'Soil_Type': 'Clay',               # Categorical            
    'Rainfall_mm': 150.5,              # Numerical
    'Temperature_Celsius': 25.3,       # Numerical
    'Fertilizer_Used': 80.0,           # Numerical
    'Irrigation_Used': 1,              # Numerical (1 = Yes, 0 = No)
    'Weather_Condition': 'Sunny',      # Categorical
    'Days_to_Harvest': 120,            # Numerical
    'Yield_tons_per_hectare': 3.5      # Numerical
}

# Encode categorical columns
for column, le in label_encoders.items():
    if column in input_data:
        input_data[column] = le.transform([input_data[column]])[0]

# Convert input_data to a DataFrame
input_features = pd.DataFrame([input_data])

# Align the input features to match the feature columns used during training
input_features = input_features.reindex(columns=feature_columns, fill_value=0)

# Scale the features
input_features_scaled = scaler.transform(input_features)

# Predict using the scaled input
obs = input_features_scaled[0]
action, _states = model.predict(obs)

# Decode the predicted crop
predicted_crop = label_encoders['Crop'].inverse_transform([action])[0]
print(f"Predicted Crop: {predicted_crop}")