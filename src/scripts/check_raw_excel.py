
import pandas as pd
import os

file_path = r'c:\Vimesh\spice-market-scout\notebooks\Pepper_Dataset.xlsx'

def check_raw_values():
    try:
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter for dates >= 2024-05-01
        mask = df['Date'] >= '2024-05-01'
        subset = df[mask].sort_values(['Date', 'Region'])
        
        print(f"Total rows after 2024-05-01: {len(subset)}")
        
        # Show sample of values
        print("\n--- Sample values after May 2024 ---")
        print(subset[['Date', 'Region', 'Grade', 'Regional_Price']].head(20))
        
        # Check if any non-null prices exist after June 2024
        post_june = df[df['Date'] >= '2024-07-01']
        non_null_count = post_june['Regional_Price'].count()
        print(f"\nNon-null prices after July 2024: {non_null_count}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_raw_values()
