import pandas as pd
df = pd.read_csv('data/processed/cinnamon_prices.csv')
print('=== Feb 2026 rows ===')
feb_data = df[df['Date'].str.startswith('2026-02')]
print(feb_data[['Date', 'Grade', 'Region', 'Regional_Price']].head(15))
print(f'\nTotal zeros in dataset: {(df["Regional_Price"] == 0).sum()}')
print(f'Zeros in Feb 2026: {(feb_data["Regional_Price"] == 0).sum()}')
