"""
Debug: test the price collector flow for Oct 2025 alba/galle
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/collectors')

from price_collector import PriceCollector

collector = PriceCollector()

# Try to get price for alba, galle, Oct 2025
print("=== Testing price_collector for alba, galle, Oct 2025 ===")
price = collector.get_price(2025, 10, 'alba', 'galle', 'regional')
print(f"Regional price: {price}")

price_nat = collector.get_price(2025, 10, 'alba', 'galle', 'national')
print(f"National price: {price_nat}")

# Check the cache
key = collector._make_key(2025, 10, 'alba', 'galle')
print(f"\nCache key: {key}")
print(f"Key in cache: {key in collector.price_cache}")
if key in collector.price_cache:
    print(f"Cache entry: {collector.price_cache[key]}")

# Also check what keys are in cache for Oct 2025 galle
print("\n=== All cache keys for 2025-10 galle ===")
for k, v in collector.price_cache.items():
    if '2025-10' in k and 'galle' in k:
        print(f"  {k}: {v}")
