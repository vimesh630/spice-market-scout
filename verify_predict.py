import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.'))

try:
    from src import forecasting_engine as engine
    
    print("Loading artifacts...")
    model = engine.load_artifacts()
    
    data_path = "data/processed/spice_prices.csv"
    print(f"Loading and enriching data from {data_path}...")
    df = engine.load_and_prepare_data(data_path)
    
    print(f"Data shape after enrichment: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    
    # Simulate API filtering
    req_grade = "ALBA"
    req_region = "Colombo"
    
    if 'Grade' in df.columns:
        df = df[df['Grade'] == req_grade]
        if 'Region' in df.columns and req_region in df['Region'].unique():
             df = df[df['Region'] == req_region]
             
        df = df.sort_values('Date')
        print(f"Filtered DF shape: {df.shape}")
        
        print("Forecasting...")
        price = engine.forecast_prices(model, df)
        print(f"SUCCESS: Predicted Price: {price}")
    else:
        print("FAILURE: 'Grade' column missing (Enrichment failed)")

except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
