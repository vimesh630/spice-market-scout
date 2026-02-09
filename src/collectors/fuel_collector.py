"""
CEYPETCO fuel price collector.
Fetches Lanka Auto Diesel prices from Ceylon Petroleum Corporation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
from typing import Optional, Dict
import re

from config import CEYPETCO_URL


# Historical fuel price data (Lanka Auto Diesel LKR per liter)
# This serves as the primary data source since CEYPETCO doesn't have a structured API
FUEL_PRICE_HISTORY = {
    '2026-02': 277,   # Feb 2026 - Rs. 2 reduction
    '2026-01': 279,
    '2025-12': 286,
    '2025-11': 286,
    '2025-10': 286,
    '2025-09': 299,
    '2025-08': 299,
    '2025-07': 305,
    '2025-06': 305,
    '2025-05': 274,
    '2025-04': 274,
    '2025-03': 286,
    '2025-02': 286,
    '2025-01': 286,
    '2024-12': 286,
    '2024-11': 286,
    '2024-10': 283,
    '2024-09': 283,
    '2024-08': 307,
    '2024-07': 317,
    '2024-06': 317,
    '2024-05': 317,
    '2024-04': 333,
    '2024-03': 363,
    '2024-02': 363,
    '2024-01': 363,
    '2023-12': 358,
    '2023-11': 329,
    '2023-10': 356,
    '2023-09': 351,
    '2023-08': 341,
    '2023-07': 306,
    '2023-06': 308,
    '2023-05': 310,
    '2023-04': 310,
    '2023-03': 325,
    '2023-02': 405,
    '2023-01': 405,
    '2022-12': 420,
    '2022-11': 420,
    '2022-10': 430,
    '2022-09': 415,
    '2022-08': 430,
    '2022-07': 430,
    '2022-06': 430,
    '2022-05': 289,
    '2022-04': 289,
    '2022-03': 176,
    '2022-02': 176,
    '2022-01': 121,
    '2021-12': 121,
    '2021-11': 111,
    '2021-10': 111,
    '2021-09': 111,
    '2021-08': 111,
    '2021-07': 111,
    '2021-06': 111,
    '2021-05': 111,
    '2021-04': 104,
    '2021-03': 104,
    '2021-02': 104,
    '2021-01': 104,
    '2020-12': 104,
    '2020-11': 104,
    '2020-10': 104,
    '2020-09': 104,
    '2020-08': 104,
    '2020-07': 104,
    '2020-06': 104,
    '2020-05': 104,
    '2020-04': 104,
    '2020-03': 104,
    '2020-02': 104,
    '2020-01': 104,
}


def _get_cache_key(year: int, month: int) -> str:
    """Generate cache key from year and month."""
    return f"{year}-{month:02d}"


def fetch_fuel_price(year: int, month: int) -> int:
    """
    Fetch Lanka Auto Diesel price for a specific month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        Fuel price in LKR per liter
    """
    cache_key = _get_cache_key(year, month)
    
    # Check historical data first
    if cache_key in FUEL_PRICE_HISTORY:
        return FUEL_PRICE_HISTORY[cache_key]
    
    # Try to scrape current price from CEYPETCO
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(CEYPETCO_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        
        # Look for Auto Diesel price
        diesel_pattern = re.compile(r'Auto\s*Diesel.*?Rs\.?\s*(\d{2,3})', re.IGNORECASE | re.DOTALL)
        match = diesel_pattern.search(text)
        
        if match:
            price = int(match.group(1))
            FUEL_PRICE_HISTORY[cache_key] = price
            return price
            
    except Exception as e:
        print(f"Error fetching fuel price: {e}")
    
    # Fallback: use most recent known price
    available_keys = sorted(FUEL_PRICE_HISTORY.keys(), reverse=True)
    if available_keys:
        return FUEL_PRICE_HISTORY[available_keys[0]]
    
    # Ultimate fallback
    return 280


def get_fuel_price_for_date(dt: date) -> int:
    """
    Get fuel price for a specific date.
    
    Args:
        dt: Date object
        
    Returns:
        Fuel price in LKR per liter
    """
    return fetch_fuel_price(dt.year, dt.month)


def update_fuel_price(year: int, month: int, price: int) -> None:
    """
    Manually update fuel price for a month.
    Useful when new prices are announced.
    
    Args:
        year: Year
        month: Month
        price: Price in LKR per liter
    """
    cache_key = _get_cache_key(year, month)
    FUEL_PRICE_HISTORY[cache_key] = price


if __name__ == "__main__":
    # Test the fuel collector
    print("=== Fuel Price Collector Test ===\n")
    
    test_cases = [
        (2026, 2),
        (2025, 9),
        (2024, 6),
        (2023, 1),
    ]
    
    for year, month in test_cases:
        price = fetch_fuel_price(year, month)
        print(f"{year}-{month:02d}: Rs. {price}/L")
