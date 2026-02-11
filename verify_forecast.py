import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import pandas as pd
import numpy as np
from src import forecasting_engine as engine
from datetime import datetime

# 1. Create dummy data mocking the schema
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=20, freq='ME'),
    'Regional_Price': np.random.uniform(100, 200, 20),
    'National_Price': np.random.uniform(110, 220, 20),
    'Grade': ['Grade1'] * 20,
    'Region': ['Region1'] * 20,
    'Is_Active_Region': [1] * 20,
    'Temperature': np.random.uniform(25, 30, 20),
    'Rainfall': np.random.uniform(100, 200, 20),
    # Add other encoded/numeric columns expected by preprocess_data/prepare_sequences
    'Grade_encoded': [0] * 20,
    'Region_encoded': [0] * 20,
    'Seasonal_Impact': np.random.uniform(0, 1, 20),
    'Local_Production_Volume': np.random.uniform(100, 200, 20),
    'Local_Export_Volume': np.random.uniform(50, 100, 20),
    'Global_Production_Volume': np.random.uniform(1000, 2000, 20),
    'Global_Consumption_Volume': np.random.uniform(1000, 2000, 20),
    'Exchange_Rate': np.random.uniform(300, 350, 20),
    'Inflation_Rate': np.random.uniform(5, 10, 20),
    'Fuel_Price': np.random.uniform(300, 400, 20),
    'Indonesia_Price_in_USD': np.random.uniform(5, 10, 20),
    'Madagascar_Price_in_USD': np.random.uniform(5, 10, 20),
    'Tanzania_Price_in_USD': np.random.uniform(5, 10, 20),
})

# Ensure data is preprocessed initially
df = engine.preprocess_data(df)

# Mock Model
class MockModel:
    def predict(self, X):
        # Return a shape (1, 1) or (N, 1) depending on input
        return np.array([[0.5]]) # Scaled prediction

# Mock Scalers
class MockScaler:
    def transform(self, X):
        return X # No op
    def inverse_transform(self, X):
        return np.array([[150.0]]) # Return constant price for test

# Patch engine's scalers
import src.forecasting_engine
src.forecasting_engine.scaler_target = MockScaler()
src.forecasting_engine.scaler_features = MockScaler()

# 2. Run forecast_multistep
print("\nRunning forecast_multistep...")
dates, prices = engine.forecast_multistep(MockModel(), df, steps=3)

print("Dates:", dates)
print("Prices:", prices)

# Verify logic
assert len(dates) == 3
assert len(prices) == 3
assert prices[0] == 150.0

print("\nVerification Successful!")
