import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
import os

def generate_mock_data():
    """
    Generates realistic mock spice price data in Long Format.
    Range: Jan 1, 2018 to Today.
    Handles sparsity: Some grades are only available in specific regions.
    """
    start_date = date(2018, 1, 1)
    end_date = date.today()
    delta = end_date - start_date
    dates = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    
    # Define valid combinations (Sparsity Logic)
    # Regions from Exagri (approximate)
    regions = ['Colombo', 'Galle', 'Matara', 'Kandy', 'Ratnapura', 'Kalutara', 'Hambantota']
    
    # Grades
    grades = ['ALBA', 'C5_Special', 'C5', 'C4', 'M5', 'M4', 'H1', 'H2', 'Quills']
    
    # Define availability map (Mocking 'real' constraints)
    # ALBA is premium, mostly in southern coastal belts (Galle, Matara)
    availability = {
        'Colombo': ['C5', 'C4', 'M5', 'H1', 'H2', 'Quills'], # Commercial hub, mix but maybe not pure Alba
        'Galle': ['ALBA', 'C5_Special', 'C5', 'M5'], # Premium
        'Matara': ['ALBA', 'C5_Special', 'C4', 'M4'],
        'Kandy': ['H1', 'H2', 'Quills', 'M5'], # Hills, different grades
        'Ratnapura': ['C4', 'M4', 'Quills'],
        'Kalutara': ['C5', 'C4', 'M5', 'H1'],
        'Hambantota': ['M5', 'M4', 'H1', 'H2']
    }
    
    records = []
    
    # Base prices for each grade to keep it realistic
    base_prices = {
        'ALBA': 3500, 'C5_Special': 2800, 'C5': 2500, 'C4': 2200,
        'M5': 2000, 'M4': 1800, 'H1': 2100, 'H2': 1900, 'Quills': 1500
    }
    
    print("Generating data...")
    
    for region in regions:
        valid_grades = availability.get(region, [])
        for grade in valid_grades:
            # Create a random walk for this series
            current_price = base_prices.get(grade, 2000)
            # Region modifier (e.g. Galle is more expensive)
            if region in ['Galle', 'Matara']:
                current_price *= 1.1
            
            series_prices = []
            for d in dates:
                # Random walk
                change = np.random.normal(0, current_price * 0.015) 
                
                # Seasonality (slight bump in certain months)
                month = d.month
                if month in [12, 1, 4]: # Holiday/New Year
                   change += current_price * 0.005
                   
                current_price += change
                series_prices.append(round(current_price, 2))
                
            # Create records
            for d, p in zip(dates, series_prices):
                records.append({
                    'Date': d,
                    'Region': region,
                    'Grade': grade,
                    'Regional_Price': p,
                    'Market_Sentiment': random.choice(['Bullish', 'Bearish', 'Neutral']) # Daily noise
                })

    df = pd.DataFrame(records)
    
    # Sort
    df = df.sort_values(['Date', 'Region', 'Grade'])
    
    # Ensure directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    output_path = 'data/processed/spice_prices.csv'
    df.to_csv(output_path, index=False)
    print(f"Mock data generated (2018-Present) and saved to {output_path}")
    print(f"Total Rows: {len(df)}")
    print(f"Sample:\n{df.head()}")

if __name__ == "__main__":
    generate_mock_data()
