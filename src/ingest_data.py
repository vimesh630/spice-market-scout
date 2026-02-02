import pandas as pd
import numpy as np
import os
import sys

def ingest_commodity(commodity, input_path, file_type='excel'):
    """
    Ingests data for a specific commodity.
    """
    print(f"\n--- Ingesting {commodity.upper()} ---")
    print(f"Reading from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return False

    try:
        if file_type == 'excel':
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
        
    print(f"Original columns: {df.columns.tolist()}")

    # 1. Standardize Date
    if 'Month' in df.columns and 'Date' not in df.columns:
        print("Renaming 'Month' to 'Date'...")
        df.rename(columns={'Month': 'Date'}, inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("Error: No 'Date' or 'Month' column found.")
        return False

    # 2. Handle missing prices (Forward/Back fill)
    if 'Regional_Price' in df.columns:
        print("Handling missing prices...")
        df = df.sort_values(['Region', 'Grade', 'Date'])
        missing_before = df['Regional_Price'].isna().sum()
        df['Regional_Price'] = df.groupby(['Region', 'Grade'])['Regional_Price'].transform(lambda x: x.ffill().bfill())
        missing_after = df['Regional_Price'].isna().sum()
        print(f"Filled {missing_before - missing_after} missing prices.")
        
        # Drop remaining empty (likely due to empty groups)
        if missing_after > 0:
            df = df.dropna(subset=['Regional_Price'])
    else:
        print("Error: 'Regional_Price' column missing.")
        return False
        
    # 3. Add default sentiment if missing
    if 'Market_Sentiment' not in df.columns:
        df['Market_Sentiment'] = 'Neutral'

    # 4. Save
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{commodity}_prices.csv')
    
    # Numeric cleanup
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
         df[col] = df[col].fillna(0) # Minimal fill for features

    df.to_csv(output_path, index=False)
    print(f"Saved {commodity} data to {output_path} ({len(df)} rows)")
    return True

if __name__ == "__main__":
    # Cinnamon
    ingest_commodity(
        'cinnamon', 
        'notebooks/Cinnamon_Dataset_New_0002.xlsx', 
        file_type='excel'
    )
    
    # Clove
    ingest_commodity(
        'clove',
        'notebooks/Clove_Dataset.csv',
        file_type='csv'
    )
