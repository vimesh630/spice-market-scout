import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.'))

try:
    from src import forecasting_engine as engine
    
    commodities = [
        {'name': 'cinnamon', 'grade': 'alba', 'region': 'galle'},
        {'name': 'clove', 'grade': 'Grade1', 'region': 'Kandy'} # Assuming Grade1/Kandy exists in Clove, or fallback
    ]
    
    # We need to check available grades/regions for Clove to be sure.
    # The Clove ingestion showed "Grade" column exists. Let's assume standard names or check CSV.
    # For robust verification, we might just load data and pick first available.
    
    for com in commodities:
        print(f"\n--- Verifying {com['name'].upper()} ---")
        model = engine.load_artifacts(com['name'])
        
        data_path = f"data/processed/{com['name']}_prices.csv"
        print(f"Loading data from {data_path}...")
        
        if not os.path.exists(data_path):
             print(f"FAILURE: Data file {data_path} not found")
             continue
             
        df = engine.load_and_prepare_data(data_path)
        
        # Pick valid grade/region from data if specified doesn't exist
        req_grade = com['grade']
        req_region = com['region']
        
        if 'Grade' in df.columns:
            available_grades = df['Grade'].unique()
            if req_grade not in available_grades:
                print(f"Warning: Grade {req_grade} not found. Using {available_grades[0]}")
                req_grade = available_grades[0]
            
            df = df[df['Grade'] == req_grade]
            
            if 'Region' in df.columns:
                available_regions = df['Region'].unique()
                if req_region not in available_regions:
                     print(f"Warning: Region {req_region} not found. Using {available_regions[0]}")
                     req_region = available_regions[0]
                     
                df = df[df['Region'] == req_region]
                
            df = df.sort_values('Date')
            print(f"Filtered DF shape: {df.shape}")
            
            print("Forecasting...")
            price = engine.forecast_prices(model, df)
            print(f"SUCCESS: Predicted Price for {com['name']} ({req_grade}/{req_region}): {price}")
        else:
            print("FAILURE: 'Grade' column missing")

except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
