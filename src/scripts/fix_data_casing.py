
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR

def fix_casing():
    file_path = os.path.join(PROCESSED_DATA_DIR, 'clove_prices.csv')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Original types:", df['Grade'].unique())
    
    # Fix Grade
    if 'Grade' in df.columns:
        df['Grade'] = df['Grade'].astype(str).str.title()
        # Ensure specific corrections if needed, e.g. 'Stem' vs 'stem' -> 'Stem'
        
    # Fix Region
    if 'Region' in df.columns:
        df['Region'] = df['Region'].astype(str).str.title()
        
    print("Fixed types:", df['Grade'].unique())
    print("Fixed regions example:", df['Region'].unique()[:5])
    
    # Save
    df.to_csv(file_path, index=False)
    print("Saved fixed dataset.")
    
    # Verify dates
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")

if __name__ == "__main__":
    fix_casing()
