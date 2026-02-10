"""
Analyze the dataset to find all unique grades and regions,
and identify which combinations are missing data after 2025-10.
"""
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/cinnamon_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=== Unique Grades in Dataset ===")
unique_grades = sorted(df['Grade'].unique())
print(unique_grades)
print(f"Total: {len(unique_grades)}")

print("\n=== Unique Regions in Dataset ===")
unique_regions = sorted(df['Region'].unique())
print(unique_regions)
print(f"Total: {len(unique_regions)}")

print("\n=== My Current Config ===")
config_grades = ['alba', 'c4', 'c5', 'c5sp', 'h1', 'h2']
config_regions = ['colombo', 'galle', 'hambantota', 'kandy', 'kurunegala', 'matara', 'ratnapura']
print(f"Config grades: {config_grades}")
print(f"Config regions: {config_regions}")

print("\n=== Missing from Config ===")
missing_grades = set(unique_grades) - set(config_grades)
missing_regions = set(unique_regions) - set(config_regions)
print(f"Missing grades: {missing_grades}")
print(f"Missing regions: {missing_regions}")

print("\n=== Combinations with data only until 2025-10 or earlier ===")
# Group by Grade, Region and find max date
latest_dates = df.groupby(['Grade', 'Region'])['Date'].max().reset_index()
latest_dates.columns = ['Grade', 'Region', 'Latest_Date']

# Filter for combinations that haven't been updated past Oct 2025
stale_combinations = latest_dates[latest_dates['Latest_Date'] < '2025-11-01']
print(stale_combinations.sort_values(['Region', 'Grade']))
print(f"\nTotal stale combinations: {len(stale_combinations)}")
