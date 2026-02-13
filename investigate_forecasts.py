import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import forecasting_engine as engine

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "investigation_output.txt"

def analyze_commodity(commodity):
    data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{commodity}_prices.csv')
    if not os.path.exists(data_path):
        print(f"[{commodity}] Data file not found.")
        return

    print(f"\n--- Analyzing {commodity} ---")
    df = engine.load_and_prepare_data(data_path)
    
    # Sort and pick last chunk
    if 'Date' in df.columns: df = df.sort_values('Date')
    
    # Pick a specific series if applicable
    if 'Grade' in df.columns: 
        grade = df['Grade'].mode()[0]
        region = df['Region'].mode()[0]
        print(f"Selecting Grade: {grade}, Region: {region}")
        df = df[(df['Grade'] == grade) & (df['Region'] == region)]
    
    last_real = df.iloc[-1]['Regional_Price']
    print(f"Last Real Price: {last_real:.2f}")

    # Load Model
    model = engine.load_artifacts(commodity)
    if not model:
        print(f"Model for {commodity} not found or failed to load. Training in-memory...")
        # Train temporary model
        df_train = engine.load_and_prepare_data(data_path)
        model, _, _ = engine.train_model(df_train, commodity=commodity, epochs=15)

    # Forecast
    scenarios = engine.forecast_multistep(model, df, steps=12, commodity=commodity)
    
    for name, data in scenarios.items():
        prices = data['prices']
        print(f"\nScenario: {name}")
        print(f"Prices: {[round(p, 2) for p in prices]}")
        
        # Analyze Shape
        if len(prices) > 0:
            pct_change = (prices[-1] - last_real) / last_real * 100
            print(f"  Total Change: {pct_change:.2f}%")
            
            # Check for reversion/dip
            if prices[0] < last_real and prices[-1] > prices[0]:
                 print("  [OBSERVATION] Dip then recovery?")
            if prices[0] < last_real * 0.95:
                 print("  [WARNING] Initial drop > 5% despite anchoring?")

if __name__ == "__main__":
    analyze_commodity('cinnamon')
    analyze_commodity('pepper')
