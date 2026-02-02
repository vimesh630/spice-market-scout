import pandas as pd
import numpy as np
import os
import sys

def ingest_user_data():
    """
    Reads the user-provided Excel dataset and converts it to the CSV format 
    expected by the forecasting engine.
    """
    input_path = 'notebooks/Cinnamon_Dataset_New_0002.xlsx'
    output_path = 'data/processed/spice_prices.csv'
    
    print(f"Reading data from {input_path}...")
    
    try:
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic validation
    # Check for 'Month' and rename to 'Date' if found (common in user dataset)
    if 'Month' in df.columns and 'Date' not in df.columns:
        print("Renaming 'Month' column to 'Date'...")
        df.rename(columns={'Month': 'Date'}, inplace=True)
        
    required_cols = ['Date', 'Region', 'Grade', 'Regional_Price']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        # Try to map if possible (e.g. if names are slightly different)
        # For now, assume strict schema based on inspection
        return

    # Date formatting
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        print("Converting Date column to datetime...")
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Handle missing prices
    # The user dataset has missing values in Regional_Price.
    # We will sort by Region, Grade, Date and Forward Fill, then Backward Fill
    print("Handling missing price values...")
    df = df.sort_values(['Region', 'Grade', 'Date'])
    
    missing_before = df['Regional_Price'].isna().sum()
    print(f"Missing prices before fill: {missing_before}")
    
    # ffill/bfill within each group
    df['Regional_Price'] = df.groupby(['Region', 'Grade'])['Regional_Price'].transform(lambda x: x.ffill().bfill())
    
    missing_after = df['Regional_Price'].isna().sum()
    print(f"Missing prices after fill: {missing_after}")
    
    # Drop rows where price is still missing (if entire series was empty)
    if missing_after > 0:
        print(f"Dropping {missing_after} rows with persistent missing prices...")
        df = df.dropna(subset=['Regional_Price'])
        
    # Data type conversion for engine compatibility
    # Ensure numeric columns are float
    numeric_cols = ['Regional_Price', 'National_Price', 'Local_Production_Volume', 'Local_Export_Volume', 
                   'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill remaining NaNs in features with 0 or mean
            df[col] = df[col].fillna(0) # Simple fill for external features
            
    # Add Market_Sentiment if missing (Engine expects it, though might not strictly use it if not encoded)
    if 'Market_Sentiment' not in df.columns:
        print("Adding default Market_Sentiment...")
        df['Market_Sentiment'] = 'Neutral'
        
    # Ensure directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed CSV
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Sample:\n{df.head()}")

if __name__ == "__main__":
    ingest_user_data()
