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

from config import EXAGRI_URL, CACHE_DIR, DATA_DIR, get_commodity_config
from collectors.exagri_scraper import get_exagri_scraper, get_scraper, fetch_regional_price_exagri, fetch_national_price_exagri


class PriceCollector:
    """
    Collects spice prices from exagri.info or cached data.
    Supports multiple commodities via the commodity parameter.
    """
    
    def __init__(self, commodity: str = 'cinnamon'):
        self.commodity = commodity
        self.price_cache: Dict[str, Dict] = {}
        self._load_cache()
    
    def _get_cache_path(self) -> Path:
        """Get path to price cache file (per-commodity)."""
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f'price_cache_{self.commodity}.json' if self.commodity != 'cinnamon' else 'price_cache.json'
        return cache_dir / filename
    
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
            scraper = get_scraper(self.commodity)
            prices = scraper.get_monthly_average_prices(year, month)
            
            count = 0
            for region, grades in prices.items():
                for grade, price in grades.items():
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
        price_type: str = 'regional',
        force_refresh: bool = False
    ) -> Optional[float]:
        """
        Get price for a specific grade, region, and month.
        
        Args:
            year: Year
            month: Month (1-12)
            grade: Cinnamon grade (e.g., 'c4', 'alba')
            region: Region name
            price_type: 'regional' or 'national'
            force_refresh: If True, skip cache and re-fetch from scraper
            
        Returns:
            Price in LKR, or None if not available
        """
        key = self._make_key(year, month, grade.lower(), region.lower())
        
        # Check cache (unless force_refresh)
        if not force_refresh and key in self.price_cache:
            data = self.price_cache[key]
            return data.get(f'{price_type}_price')
        
        # Fetch from scraper
        if force_refresh:
            print(f"Force refreshing {key} from scraper...")
        else:
            print(f"Cache miss for {key}, fetching from scraper...")
        
        if self._fetch_month_from_scraper(year, month) > 0:
            # Try getting from cache again
            if key in self.price_cache:
                data = self.price_cache[key]
                return data.get(f'{price_type}_price')
        
        # Fallback to individual fetch if bulk fetch failed or didn't contain our key
        # (This is legacy support, mostly unnecessary if bulk fetch works)
        if price_type == 'regional':
             price = fetch_regional_price_exagri(
                region, grade, year, month, commodity=self.commodity
            )
        else:
             price = fetch_national_price_exagri(
                grade, year, month, commodity=self.commodity
            )
            
        if price:
             # Cache this individual result
             existing = self.price_cache.get(key, {})
             existing[f'{price_type}_price'] = price
             existing['updated_at'] = datetime.now().isoformat()
             self.price_cache[key] = existing
             self._save_cache()
             return price

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
    
    def import_from_existing_dataset(self, commodity: str) -> int:
        """
        Import price data from the existing processed dataset.
        This bootstraps the cache with historical data.
        """
        # Ensure we use the commodity passed, or self.commodity
        if not commodity:
            commodity = self.commodity
            
        config = get_commodity_config(commodity)
        dataset_path = Path(DATA_DIR) / 'processed' / config['data_file']
        
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
                        regional_price = float(row['Regional_Price'])
                        # National_Price may not exist (e.g., pepper dataset)
                        national_price_str = row.get('National_Price', '')
                        national_price = float(national_price_str) if national_price_str else regional_price
                        self.set_price(
                            year=dt.year,
                            month=dt.month,
                            grade=row['Grade'],
                            region=row['Region'],
                            regional_price=regional_price,
                            national_price=national_price
                        )
                        count += 1
                    except (KeyError, ValueError) as e:
                        continue
                        
        except Exception as e:
            print(f"Error importing from dataset: {e}")
            
        print(f"Imported {count} price records from {commodity} dataset")
        return count


# Singleton instance
_collector: Optional[PriceCollector] = None


def get_price_collector(commodity: str = 'cinnamon') -> PriceCollector:
    """Get the singleton price collector instance."""
    global _collector
    # If collector logic needs to change based on commodity, we might need a dict of collectors
    # For now, let's assume we re-instantiate if commodity changes or just keep one active
    if _collector is None or _collector.commodity != commodity:
        _collector = PriceCollector(commodity)
    return _collector
