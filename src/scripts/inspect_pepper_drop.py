
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, COMMODITY_CONFIG

def inspect_pepper_drop():
    input_file = os.path.join(PROCESSED_DATA_DIR, 'pepper_prices.csv')
    
    if not os.path.exists(input_file):
        print("File not found.")
        return

    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for dates around May 2024
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    subset = df[mask].sort_values(['Date', 'Region', 'Grade'])
    
    # Group by Date and Grade to see average price trends
    trends = subset.groupby(['Date', 'Grade'])['Regional_Price'].mean().reset_index()
    print("\n--- Average Price Trend (2024) ---")
    print(trends)
    
    # Check specific transition
    print("\n--- Detailed View around 2024-05-01 ---")
    transition_mask = (df['Date'] >= '2024-04-01') & (df['Date'] <= '2024-06-01')
    print(df[transition_mask].groupby(['Date', 'Grade'])['Regional_Price'].mean())

if __name__ == "__main__":
    inspect_pepper_drop()
