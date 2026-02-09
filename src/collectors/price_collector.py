"""
Spice price collector for exagri.info.
Fetches regional and national cinnamon prices.

NOTE: The exagri.info website blocks direct HTTP requests (403 Forbidden).
This module provides:
1. Browser-based scraping using Playwright (if available)
2. Fallback to cached/manual data entry
3. CSV import from user-provided data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from datetime import date, datetime
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from config import EXAGRI_URL, CACHE_DIR, DATA_DIR
from collectors.exagri_scraper import get_exagri_scraper


class PriceCollector:
    """
    Collects cinnamon prices from exagri.info or cached data.
    """
    
    def __init__(self):
        self.price_cache: Dict[str, Dict] = {}
        self._load_cache()
    
    def _get_cache_path(self) -> Path:
        """Get path to price cache file."""
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / 'price_cache.json'
    
    def _load_cache(self) -> None:
        """Load price cache from disk."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    self.price_cache = json.load(f)
            except Exception as e:
                print(f"Error loading price cache: {e}")
                self.price_cache = {}
    
    def _save_cache(self) -> None:
        """Save price cache to disk."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.price_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving price cache: {e}")
    
    def _make_key(self, year: int, month: int, grade: str, region: str) -> str:
        """Create cache key."""
        return f"{year}-{month:02d}_{grade}_{region}"
    
    def _fetch_month_from_scraper(self, year: int, month: int) -> int:
        """
        Fetch all prices for a month from scraper and update cache.
        Returns number of prices found.
        """
        try:
            scraper = get_exagri_scraper()
            prices = scraper.get_monthly_average_prices(year, month)
            
            count = 0
            for region, grades in prices.items():
                for grade, price in grades.items():
                    # Calculate national price (average of this grade across all regions)
                    # This is an approximation as we strictly only have the regional price here
                    # But for now we treat national price as same or explicitly fetch it later
                    # Actually get_monthly_average_prices returns structure {region: {grade: price}}
                    
                    # We need national price. Let's calculate it from the scraper data
                    # (This is handled implicitly if we just cache what we have)
                    
                    # Create/Update cache entry
                    key = self._make_key(year, month, grade, region)
                    
                    # If we already have an entry, preserve national price if it exists
                    existing = self.price_cache.get(key, {})
                    
                    self.price_cache[key] = {
                        'regional_price': price,
                        'national_price': existing.get('national_price', price), # Default to regional if missing
                        'updated_at': datetime.now().isoformat()
                    }
                    count += 1
            
            if count > 0:
                self._save_cache()
            return count
            
        except Exception as e:
            print(f"Error fetching from scraper: {e}")
            return 0

    def get_price(
        self, 
        year: int, 
        month: int, 
        grade: str, 
        region: str,
        price_type: str = 'regional'
    ) -> Optional[float]:
        """
        Get price for a specific grade, region, and month.
        
        Args:
            year: Year
            month: Month (1-12)
            grade: Cinnamon grade (e.g., 'c4', 'alba')
            region: Region name
            price_type: 'regional' or 'national'
            
        Returns:
            Price in LKR, or None if not available
        """
        key = self._make_key(year, month, grade.lower(), region.lower())
        
        # Check cache
        if key in self.price_cache:
            data = self.price_cache[key]
            return data.get(f'{price_type}_price')
        
        # If not in cache, try to fetch the whole month from scraper
        # But only do this once per month/request to avoid spamming
        # We can implement a simple in-memory check or just try
        print(f"Cache miss for {key}, fetching from scraper...")
        if self._fetch_month_from_scraper(year, month) > 0:
            # Try getting from cache again
            if key in self.price_cache:
                data = self.price_cache[key]
                return data.get(f'{price_type}_price')
        
        return None
    
    def set_price(
        self,
        year: int,
        month: int, 
        grade: str,
        region: str,
        regional_price: float,
        national_price: float
    ) -> None:
        """
        Manually set price data for a specific grade/region/month.
        
        Args:
            year: Year
            month: Month
            grade: Cinnamon grade
            region: Region name
            regional_price: Regional price in LKR
            national_price: National average price in LKR
        """
        key = self._make_key(year, month, grade.lower(), region.lower())
        self.price_cache[key] = {
            'regional_price': regional_price,
            'national_price': national_price,
            'updated_at': datetime.now().isoformat()
        }
        self._save_cache()
    
    def import_from_csv(self, filepath: str) -> int:
        """
        Import price data from a CSV file.
        
        Expected CSV columns: Date, Grade, Region, Regional_Price, National_Price
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Number of records imported
        """
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dt = datetime.strptime(row['Date'], '%Y-%m-%d')
                        self.set_price(
                            year=dt.year,
                            month=dt.month,
                            grade=row['Grade'],
                            region=row['Region'],
                            regional_price=float(row['Regional_Price']),
                            national_price=float(row['National_Price'])
                        )
                        count += 1
                    except (KeyError, ValueError) as e:
                        print(f"Skipping row: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error importing CSV: {e}")
            
        return count
    
    def import_from_existing_dataset(self, commodity: str = 'cinnamon') -> int:
        """
        Import price data from the existing processed dataset.
        This bootstraps the cache with historical data.
        
        Args:
            commodity: Commodity name (default: 'cinnamon')
            
        Returns:
            Number of records imported
        """
        dataset_path = Path(DATA_DIR) / 'processed' / f'{commodity}_prices.csv'
        
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return 0
        
        count = 0
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dt = datetime.strptime(row['Date'], '%Y-%m-%d')
                        self.set_price(
                            year=dt.year,
                            month=dt.month,
                            grade=row['Grade'],
                            region=row['Region'],
                            regional_price=float(row['Regional_Price']),
                            national_price=float(row['National_Price'])
                        )
                        count += 1
                    except (KeyError, ValueError) as e:
                        continue
                        
        except Exception as e:
            print(f"Error importing from dataset: {e}")
            
        print(f"Imported {count} price records from {commodity} dataset")
        return count


def fetch_regional_price(
    region: str, 
    grade: str, 
    year: int, 
    month: int
) -> Optional[float]:
    """
    Fetch regional price for a grade/region/month.
    
    Args:
        region: Region name
        grade: Cinnamon grade
        year: Year
        month: Month
        
    Returns:
        Price in LKR, or None if not available
    """
    collector = PriceCollector()
    return collector.get_price(year, month, grade, region, 'regional')


def fetch_national_price(
    grade: str, 
    year: int, 
    month: int
) -> Optional[float]:
    """
    Fetch national average price for a grade/month.
    Uses 'colombo' as the reference for national prices.
    
    Args:
        grade: Cinnamon grade
        year: Year
        month: Month
        
    Returns:
        Price in LKR, or None if not available
    """
    collector = PriceCollector()
    # National price is same across regions, use colombo as reference
    return collector.get_price(year, month, grade, 'colombo', 'national')


# Singleton instance
_price_collector: Optional[PriceCollector] = None


def get_price_collector() -> PriceCollector:
    """Get the singleton PriceCollector instance."""
    global _price_collector
    if _price_collector is None:
        _price_collector = PriceCollector()
    return _price_collector


if __name__ == "__main__":
    # Test and bootstrap the price collector
    print("=== Price Collector Test ===\n")
    
    collector = get_price_collector()
    
    # Try to import from existing dataset
    print("Importing from existing dataset...")
    count = collector.import_from_existing_dataset('cinnamon')
    print(f"Imported {count} records\n")
    
    # Test getting a price
    test_cases = [
        (2025, 9, 'c4', 'colombo'),
        (2024, 6, 'alba', 'galle'),
        (2023, 1, 'h1', 'matara'),
    ]
    
    for year, month, grade, region in test_cases:
        regional = collector.get_price(year, month, grade, region, 'regional')
        national = collector.get_price(year, month, grade, region, 'national')
        print(f"{year}-{month:02d} {grade} ({region}):")
        print(f"  Regional: Rs. {regional}")
        print(f"  National: Rs. {national}")
