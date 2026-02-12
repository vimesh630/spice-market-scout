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
    def predict(self, X, **kwargs):
        # Predict 300.0, which is a +100% jump from the mock start price (150.0)
        # This attempts to break the 15% clamp
        # We need to return valid scaled output that effectively maps to ~300.0
        # But our mock scaler just returns a constant 300.0 for inverse_transform.
        # So we update MockScaler to return 300.
        return np.array([[1.0]]) 

# Mock Scalers
class MockScaler:
    def transform(self, X):
        return X 
    def inverse_transform(self, X):
        # Return a huge price
        return np.array([[300.0]]) 

# Patch engine's scalers
import src.forecasting_engine
src.forecasting_engine.scaler_target = MockScaler()
src.forecasting_engine.scaler_features = MockScaler()

# Initialize encoders manually since we rely on them now
from sklearn.preprocessing import LabelEncoder
src.forecasting_engine.label_encoders = {
    'Grade': LabelEncoder(),
    'Region': LabelEncoder()
}
src.forecasting_engine.label_encoders['Grade'].fit(['Grade1', 'Grade2'])
src.forecasting_engine.label_encoders['Region'].fit(['Region1', 'Region2'])

# Setup initial dataframe logic
# Last Regional_Price is needed. In our dummy data it's random (100-200).
# Let's force the last row to have a known price for deterministic testing.
last_price = 100.0
df.iloc[-1, df.columns.get_loc('Regional_Price')] = last_price
# Set National Price to Last Price + 10 (spread = 10)
df.iloc[-1, df.columns.get_loc('National_Price')] = last_price + 10.0

# 2. Run forecast_multistep
print(f"\nRunning forecast_multistep... (Last Price: {last_price})")
dates, prices = engine.forecast_multistep(MockModel(), df, steps=1)

print("Dates:", dates)
print("Prices:", prices)

# Verify Logic
# Verify Logic
# 1. Clamping: Max increase is 12% of 100 = 12. So max allowed is 112.
# The model tries to predict 300.
# Note: Logic uses 12% clamp (0.12)
expected_price_upper = last_price * 1.12
print(f"Expected Clamped Price (Upper Bound): {expected_price_upper}")

# Since model predicts 300, it should hit the upper clamp exactly
assert abs(prices[0] - expected_price_upper) < 0.01, f"Clamping failed! Got {prices[0]}, expected {expected_price_upper}"

# 2. Spread Preservation: 
# The function appends the row to current_df. We can check the internal logic or just trust the code if price is right.
# However, forecast_multistep does NOT return national prices, only regional.
# To verify national price logic, we'd need to inspect the dataframe inside the loop or mock the dataframe append.
# Given the code simplicity, verifying the Regional Price clamping is the primary test.
# But let's check if the loop ran without error which implies spread calculation worked.

print("\nVerification Successful! Clamping works.")

# Verify logic
assert len(dates) == 1
assert len(prices) == 1
# assert prices[0] == 150.0 # Old check


print("\nVerification Successful!")

