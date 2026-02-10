import pandas as pd
df = pd.read_csv('data/processed/cinnamon_prices.csv')
alba_galle = df[(df['Grade']=='alba') & (df['Region']=='galle')].sort_values('Date')
print("=== Alba, Galle (FINAL - after force-refresh) ===")
print(alba_galle[['Date','Regional_Price']].tail(10).to_string(index=False))

print("\n=== Galle, Oct 2025 all grades ===")
oct_galle = df[(df['Region']=='galle') & (df['Date'].str.startswith('2025-10'))].sort_values('Grade')
print(oct_galle[['Date','Grade','Regional_Price']].to_string(index=False))

print(f"\nTotal zeros: {(df['Regional_Price'] <= 0).sum()}")
print(f"Total rows: {len(df)}")
