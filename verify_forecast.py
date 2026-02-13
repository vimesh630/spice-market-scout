import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import pandas as pd
import numpy as np
from src import forecasting_engine as engine
from datetime import datetime
import json

# 1. Create dummy fixture (No Randomness)
dates = pd.date_range(start='2024-01-01', periods=24, freq='ME')
regional_prices = np.linspace(100, 124, 24) # Steady linear growth
national_prices = regional_prices * 1.1

df = pd.DataFrame({
    'Date': dates,
    'Regional_Price': regional_prices,
    'National_Price': national_prices,
    'Grade': ['Grade1'] * 24,
    'Region': ['Region1'] * 24,
    'Temperature': [25.0] * 24,
    'Rainfall': [100.0] * 24,
    'Exchange_Rate': [300.0] * 24,
    'Inflation_Rate': [5.0] * 24,
    'Fuel_Price': [350.0] * 24,
    'Local_Production_Volume': [150.0] * 24,
    'Is_Active_Region': [1] * 24,
})

# Mock Feature Cols
feature_cols = ['Regional_Price', 'Temperature'] # Minimal set for testing
os.makedirs('models/lstm_grade1', exist_ok=True)
with open('models/lstm_grade1/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

# Mock Model
class MockModel:
    def predict(self, X, **kwargs):
        # Always predict a massive jump to test clamping
        # Returns scaled value 10.0 (assuming scaler maps 0-1 to reasonable range)
        return np.array([[10.0]]) 

# Mock Scalers
class MockScalerFeatures:
    def transform(self, X):
         # Expecting 2 features
         if X.shape[1] != 2:
             raise ValueError(f"Shape mismatch: {X.shape}")
         return X # Pass through

class MockScalerTarget:
    def fit_transform(self, X): return X
    def inverse_transform(self, X):
        # We simulate the model predicting "500.0"
        return np.array([[500.0]]) 

# Patch engine
engine.scaler_target = MockScalerTarget()
engine.scaler_features = MockScalerFeatures()
# Patch label encoders
from sklearn.preprocessing import LabelEncoder
engine.label_encoders = {'Grade': LabelEncoder(), 'Region': LabelEncoder()}
engine.label_encoders['Grade'].fit(['Grade1'])
engine.label_encoders['Region'].fit(['Region1'])

# Preprocess
df = engine.preprocess_data(df)

# TEST: forecast_multistep
print("\n--- Testing Phase 3 Forecast Logic ---")
last_price = df.iloc[-1]['Regional_Price']
print(f"Last Price: {last_price}")

try:
    # We pass 'grade1' as commodity to trigger model dir search
    scenarios = engine.forecast_multistep(MockModel(), df, steps=3, commodity='grade1')
    
    print("\nScenarios Generated:")
    for name, data in scenarios.items():
        print(f"\n{name}:")
        print(f"Dates: {data['dates']}")
        print(f"Prices: {data['prices']}")
        
        # VERIFY CLAMPING
        # Volatility of linear series is low.
        # The adaptive clamp should be tight.
        # But we also have a hard safety cap of +/- 20% likely (or whatever logic we implement)
        # Verify it didn't jump to 500.0
        assert data['prices'][0] < 500.0, f"{name} Scenario failed to clamp! Got {data['prices'][0]}"
        assert data['prices'][0] > last_price, "Should have increased (clamped up)"
        
    print("\n✅ Verification Successful: Scenarios generated and Clamping active.")

except Exception as e:
    print(f"\n❌ Verification Failed: {e}")
    import traceback
    traceback.print_exc()
