"""Inspect the Pepper Dataset"""
import pandas as pd

# Read the Excel file
df = pd.read_excel('notebooks/Pepper_Dataset.xlsx')

print("=== Pepper Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
print(f"\nBasic stats:")
print(df.describe())

# Check unique values for categorical columns
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        print(f"\n{col} unique values ({df[col].nunique()}): {sorted(df[col].unique())[:20]}")

# Check date range
if 'Date' in df.columns:
    print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
    
# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nZero values in numeric columns:")
for col in df.select_dtypes(include=['number']).columns:
    print(f"  {col}: {(df[col] == 0).sum()} zeros")
