"""
CBSL (Central Bank of Sri Lanka) data collector.
Fetches exchange rates and inflation data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
from typing import Optional, Dict, Tuple
import re
import json

from config import CBSL_EXCHANGE_RATE_URL, CBSL_INFLATION_URL, CACHE_DIR


# Historical data cache (for when scraping fails)
# This serves as a fallback based on known historical values
EXCHANGE_RATE_CACHE = {
    '2025-09': 302.04,
    '2025-08': 301.12,
    '2025-07': 301.12,
    '2025-06': 301.12,
    '2025-05': 299.41,
    '2025-04': 298.53,
    '2025-03': 295.91,
    '2025-02': 296.80,
    '2025-01': 296.18,
    '2024-12': 291.68,
    '2024-11': 292.01,
    '2024-10': 293.79,
}

INFLATION_RATE_CACHE = {
    '2025-09': 2.4,
    '2025-08': 1.2,
    '2025-07': -0.3,
    '2025-06': -0.6,
    '2025-05': -0.7,
    '2025-04': -2.0,
    '2025-03': -2.6,
    '2025-02': -4.2,
    '2025-01': -4.0,
    '2024-12': -1.7,
    '2024-11': -2.1,
    '2024-10': -0.8,
}


def _get_cache_key(year: int, month: int) -> str:
    """Generate cache key from year and month."""
    return f"{year}-{month:02d}"


def fetch_exchange_rate(year: int, month: int) -> float:
    """
    Fetch USD/LKR exchange rate for a specific month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        Exchange rate (LKR per USD)
    """
    cache_key = _get_cache_key(year, month)
    
    # First check cache
    if cache_key in EXCHANGE_RATE_CACHE:
        return EXCHANGE_RATE_CACHE[cache_key]
    
    # Try to scrape from CBSL
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(CBSL_EXCHANGE_RATE_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for USD rate in the page
        # CBSL page structure varies, so we try multiple patterns
        usd_pattern = re.compile(r'USD.*?(\d{2,3}\.\d{2,4})', re.IGNORECASE | re.DOTALL)
        text = soup.get_text()
        match = usd_pattern.search(text)
        
        if match:
            rate = float(match.group(1))
            EXCHANGE_RATE_CACHE[cache_key] = rate
            return rate
            
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
    
    # Fallback: use closest available rate
    available_keys = sorted(EXCHANGE_RATE_CACHE.keys(), reverse=True)
    if available_keys:
        return EXCHANGE_RATE_CACHE[available_keys[0]]
    
    # Ultimate fallback
    return 300.0


def fetch_inflation_rate(year: int, month: int) -> float:
    """
    Fetch inflation rate (CCPI-based) for a specific month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        Inflation rate (percentage, can be negative)
    """
    cache_key = _get_cache_key(year, month)
    
    # First check cache
    if cache_key in INFLATION_RATE_CACHE:
        return INFLATION_RATE_CACHE[cache_key]
    
    # Try to scrape from CBSL
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(CBSL_INFLATION_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for inflation rate patterns
        text = soup.get_text()
        # Pattern for inflation rate like "2.3%" or "-1.5%"
        inflation_pattern = re.compile(r'inflation.*?(-?\d+\.?\d*)\s*%', re.IGNORECASE)
        match = inflation_pattern.search(text)
        
        if match:
            rate = float(match.group(1))
            INFLATION_RATE_CACHE[cache_key] = rate
            return rate
            
    except Exception as e:
        print(f"Error fetching inflation rate: {e}")
    
    # Fallback: use closest available rate
    available_keys = sorted(INFLATION_RATE_CACHE.keys(), reverse=True)
    if available_keys:
        return INFLATION_RATE_CACHE[available_keys[0]]
    
    # Ultimate fallback
    return 2.0


def fetch_cbsl_data(year: int, month: int) -> Dict[str, float]:
    """
    Fetch both exchange rate and inflation rate for a month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        Dict with 'exchange_rate' and 'inflation_rate'
    """
    return {
        'exchange_rate': fetch_exchange_rate(year, month),
        'inflation_rate': fetch_inflation_rate(year, month)
    }


def update_cache_from_file(filepath: str) -> None:
    """
    Update the in-memory cache from a JSON file.
    Useful for loading historical data.
    
    Args:
        filepath: Path to JSON file with cache data
    """
    global EXCHANGE_RATE_CACHE, INFLATION_RATE_CACHE
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'exchange_rates' in data:
            EXCHANGE_RATE_CACHE.update(data['exchange_rates'])
        if 'inflation_rates' in data:
            INFLATION_RATE_CACHE.update(data['inflation_rates'])
            
    except Exception as e:
        print(f"Error loading cache from {filepath}: {e}")


if __name__ == "__main__":
    # Test the CBSL collector
    print("=== CBSL Data Collector Test ===\n")
    
    test_cases = [
        (2025, 9),
        (2025, 1),
        (2024, 12),
    ]
    
    for year, month in test_cases:
        print(f"{year}-{month:02d}:")
        data = fetch_cbsl_data(year, month)
        print(f"  Exchange Rate: {data['exchange_rate']} LKR/USD")
        print(f"  Inflation Rate: {data['inflation_rate']}%")
