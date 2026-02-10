"""
Script to fix zero prices in the dataset by forward-filling from the last known price.
This addresses the issue where recent data updates inserted 0.0 for missing prices.
"""
import pandas as pd
import os

# Path to the dataset
data_path = 'data/processed/cinnamon_prices.csv'
backup_path = 'data/processed/cinnamon_prices_backup.csv'

print("=== Fixing Zero Prices in Dataset ===")

# Load dataset
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows")

# Count zeros before fix
zeros_before = (df['Regional_Price'] == 0).sum()
print(f"Zeros before fix: {zeros_before}")

# Backup original
df.to_csv(backup_path, index=False)
print(f"Backup saved to {backup_path}")

# Sort by Region, Grade, Date for proper forward fill
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Region', 'Grade', 'Date'])

# Replace 0 with NaN then forward fill within each group
df['Regional_Price'] = df['Regional_Price'].replace(0, pd.NA)
df['Regional_Price'] = df.groupby(['Region', 'Grade'])['Regional_Price'].transform(lambda x: x.ffill())

# Also fix National_Price if it has zeros
df['National_Price'] = df['National_Price'].replace(0, pd.NA)
df['National_Price'] = df.groupby(['Region', 'Grade'])['National_Price'].transform(lambda x: x.ffill())

# Fill any remaining NaN with a default (shouldn't happen for valid combinations)
df['Regional_Price'] = df['Regional_Price'].fillna(3000.0)
df['National_Price'] = df['National_Price'].fillna(3000.0)

# Count zeros after fix
zeros_after = (df['Regional_Price'] == 0).sum()
print(f"Zeros after fix: {zeros_after}")

# Save fixed dataset
df.to_csv(data_path, index=False)
print(f"Fixed dataset saved to {data_path}")

# Show Feb 2026 data after fix
print("\n=== Feb 2026 data after fix ===")
feb_data = df[df['Date'].dt.month == 2]
feb_data = feb_data[feb_data['Date'].dt.year == 2026]
print(feb_data[['Date', 'Grade', 'Region', 'Regional_Price']].head(15))
