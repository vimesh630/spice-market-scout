"""
Convert Pepper_Dataset.xlsx to a normalized CSV for the forecasting pipeline.
- Normalize grade/region names to lowercase
- Forward-fill missing Regional_Price within each grade/region group
- Drop entirely empty columns
- Save to data/processed/pepper_prices.csv
"""
import pandas as pd
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, 'notebooks', 'Pepper_Dataset.xlsx')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'data', 'processed', 'pepper_prices.csv')

print("=== Pepper Dataset Conversion ===")
print(f"Input:  {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# 1. Read Excel
df = pd.read_excel(INPUT_PATH)
print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns")

# 2. Drop entirely empty columns
empty_cols = [col for col in df.columns if df[col].isna().all()]
if empty_cols:
    print(f"Dropping empty columns: {empty_cols}")
    df = df.drop(columns=empty_cols)

# 3. Normalize grade and region names to lowercase
df['Grade'] = df['Grade'].str.lower().str.strip()
df['Region'] = df['Region'].str.lower().str.strip()

# Normalize specific region names
region_mapping = {
    'nuwaraeliya': 'nuwaraeliya',
    'nuwara_eliya': 'nuwaraeliya',
    'nuwara eliya': 'nuwaraeliya',
}
df['Region'] = df['Region'].replace(region_mapping)

print(f"\nGrades: {sorted(df['Grade'].unique())}")
print(f"Regions: {sorted(df['Region'].unique())}")

# 4. Sort by Region, Grade, Date for proper forward-fill
df = df.sort_values(['Region', 'Grade', 'Date']).reset_index(drop=True)

# 5. Forward-fill missing Regional_Price within each Grade/Region group
missing_before = df['Regional_Price'].isna().sum()
print(f"\nMissing prices before fill: {missing_before}")

df['Regional_Price'] = df.groupby(['Region', 'Grade'])['Regional_Price'].transform(
    lambda x: x.ffill().bfill()
)

missing_after = df['Regional_Price'].isna().sum()
print(f"Missing prices after fill: {missing_after}")

# Also fill Exchange_Rate and Inflation_Rate if missing
for col in ['Exchange_Rate', 'Inflation_Rate']:
    if col in df.columns and df[col].isna().any():
        missing = df[col].isna().sum()
        df[col] = df[col].ffill().bfill()
        print(f"Filled {missing} missing {col} values")

# 6. Ensure Date is formatted consistently
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# 7. Sort final output
df = df.sort_values(['Date', 'Grade', 'Region']).reset_index(drop=True)

# 8. Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n=== Conversion Complete ===")
print(f"Output: {OUTPUT_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Zero prices: {(df['Regional_Price'] <= 0).sum()}")
print(f"Missing prices: {df['Regional_Price'].isna().sum()}")

# Summary per grade/region
print(f"\nEntries per grade:")
print(df.groupby('Grade')['Regional_Price'].count().to_string())
print(f"\nEntries per region:")
print(df.groupby('Region')['Regional_Price'].count().to_string())
