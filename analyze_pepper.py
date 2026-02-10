
import pandas as pd
import os

file_path = r'c:\Vimesh\spice-market-scout\notebooks\Pepper_Dataset.xlsx'

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nUnique values in categorical columns (if any):")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].unique()[:10]}")
except Exception as e:
    print(f"Error reading file: {e}")
