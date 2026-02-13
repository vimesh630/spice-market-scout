import sys
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from src import forecasting_engine as engine

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cinnamon_prices.csv')

def analyze():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Check data content
    print("Data Head:")
    print(df.head())
    print("\nData Tail:")
    print(df.tail())

    # Load Model
    print("\nLoading Model...")
    model = engine.load_artifacts('cinnamon')
    if model is None:
        print("Model not found. Please train first.")
        # Attempt to train a quick one if missing?
        print("Training temporary model for analysis...")
        df_train = engine.load_and_prepare_data(DATA_PATH)
        model, _, _ = engine.train_model(df_train, commodity='cinnamon', epochs=10)

    # Prepare Data for Forecast
    # We'll take the real data up to the last point
    df_prepared = engine.load_and_prepare_data(DATA_PATH)
    
    # Filter for a specific series if needed (Grade/Region)
    if 'Grade' in df_prepared.columns and 'Region' in df_prepared.columns:
        # Pick the most populated one
        counts = df_prepared.groupby(['Grade', 'Region']).size()
        if not counts.empty:
            best_pair = counts.idxmax()
            print(f"Analyzing Series: {best_pair}")
            df_subset = df_prepared[(df_prepared['Grade'] == best_pair[0]) & (df_prepared['Region'] == best_pair[1])].copy()
        else:
            df_subset = df_prepared.copy()
    else:
        df_subset = df_prepared.copy()

    df_subset = df_subset.sort_values('Date')
    
    # Run Forecast
    print("\nGenerating Forecast...")
    steps = 12
    scenarios = engine.forecast_multistep(model, df_subset, steps=steps, commodity='cinnamon')
    
    # Analyze Results
    last_real_price = df_subset.iloc[-1]['Regional_Price']
    print(f"\nLast Real Price: {last_real_price:.2f}")
    
    for name, data in scenarios.items():
        prices = data['prices']
        print(f"\nScenario: {name}")
        print(f"Prices: {[round(p, 2) for p in prices]}")
        
        # Metrics
        start_price = prices[0]
        end_price = prices[-1]
        change_pct = (end_price - last_real_price) / last_real_price * 100
        std_dev = np.std(prices)
        
        print(f"  Change over {steps} months: {change_pct:.2f}%")
        print(f"  Std Dev (Volatility): {std_dev:.2f}")
        
        if std_dev < 1.0:
            print("  [WARNING] Forecast looks essentially FLAT.")
        
        if abs(change_pct) > 50:
             print("  [WARNING] Forecast implies extreme change (>50%).")

if __name__ == "__main__":
    analyze()
