"""
Convert notebook datasets to data/processed format for improved training.
The notebook datasets are richer/pre-filled and achieved R²~0.95.
"""
import pandas as pd
import os

def main():
    # Convert Cinnamon notebook dataset
    print('Converting Cinnamon dataset...')
    df = pd.read_excel('notebooks/Cinnamon_Dataset_New_0002.xlsx')
    print(f'  Original shape: {df.shape}')
    print(f'  Columns: {df.columns.tolist()}')
    
    # Check if 'Month' column exists (notebooks use 'Month', engine uses 'Date')
    if 'Month' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'Month': 'Date'})
        print('  Renamed Month -> Date')
    
    # Save
    out_path = 'data/processed/cinnamon_prices.csv'
    df.to_csv(out_path, index=False)
    print(f'  Saved to {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)')
    
    print()
    
    # Convert Clove notebook dataset
    print('Converting Clove dataset...')
    df2 = pd.read_csv('notebooks/Clove_Dataset.csv')
    print(f'  Original shape: {df2.shape}')
    print(f'  Columns: {df2.columns.tolist()}')
    
    if 'Month' in df2.columns and 'Date' not in df2.columns:
        df2 = df2.rename(columns={'Month': 'Date'})
        print('  Renamed Month -> Date')
    
    # Save
    out_path2 = 'data/processed/clove_prices.csv'
    df2.to_csv(out_path2, index=False)
    print(f'  Saved to {out_path2} ({df2.shape[0]} rows, {df2.shape[1]} columns)')
    
    print()
    print('Done! Both datasets converted successfully.')

if __name__ == '__main__':
    main()
