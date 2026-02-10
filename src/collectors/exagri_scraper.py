"""
Exagri.info price scraper for cinnamon prices.
Fetches weekly prices from exagri.info and calculates monthly averages.

The website has weekly price reports in format: https://exagri.info/mkt/YYYY/DD.MM.YYYY.html
Each page contains cinnamon prices in Table 6 with grades: Alba, C-5 Sp, C-5, C-4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import json
from pathlib import Path
from collections import defaultdict

from config import CACHE_DIR

# Base URL for exagri
BASE_URL = "https://exagri.info/mkt"

# HTTP headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Grade mapping from website names to our dataset names
GRADE_MAPPING = {
    'alba': 'alba',
    'c-5 sp': 'c5sp',
    'c-5': 'c5',
    'c-4': 'c4',
    'm-5': 'h1',  # M-5 maps to H1
    'm-4': 'h2',  # M-4 maps to H2
    'h-1': 'h1',
    'h-2': 'h2',
    'h-faq': 'h_faq',
    'h faq': 'h_faq',
    'faq': 'h_faq',
}

# Region mapping from website names to our dataset names
REGION_MAPPING = {
    'colombo': 'colombo',
    'galle': 'galle',
    'matara': 'matara',
    'hambantota': 'hambantota',
    'ratnapura': 'ratnapura',  # Sometimes included
    'kurunegala': 'kurunegala',
    'kandy': 'kandy',
    'badulla': 'badulla',
    'kalutara': 'kalutara',
}


class ExagriPriceScraper:
    """Scraper for exagri.info spice prices."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache_dir = Path(CACHE_DIR) / 'exagri'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._weekly_links_cache = None
    
    def get_all_weekly_links(self) -> List[Dict]:
        """
        Fetch all weekly price links from the index page.
        
        Returns:
            List of dicts with 'href', 'date', 'year', 'month' keys
        """
        if self._weekly_links_cache:
            return self._weekly_links_cache
        
        url = f"{BASE_URL}/index.html"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching index: {e}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            # Match pattern like 2026/03.02.2026.html (YYYY/DD.MM.YYYY.html)
            # The filename is: DD.MM.YYYY.html
            match = re.search(r'(\d{4})/(\d{2})\.(\d{2})\.(\d{4})\.html', href)
            if match:
                folder_year = int(match.group(1))
                day = int(match.group(2))
                month = int(match.group(3))
                year = int(match.group(4))
                try:
                    dt = date(year, month, day)
                    links.append({
                        'href': href,
                        'date': dt,
                        'year': year,
                        'month': month,
                        'day': day,
                    })
                except ValueError:
                    # Invalid date, skip
                    continue
        
        # Sort by date
        links.sort(key=lambda x: x['date'], reverse=True)
        self._weekly_links_cache = links
        return links
    
    def get_weekly_links_for_month(self, year: int, month: int) -> List[Dict]:
        """
        Get weekly links for a specific month.
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            List of weekly links for that month
        """
        all_links = self.get_all_weekly_links()
        return [
            link for link in all_links 
            if link['year'] == year and link['month'] == month
        ]
    
    def fetch_weekly_prices(self, href: str) -> Dict[str, Dict[str, float]]:
        """
        Fetch prices from a weekly price page.
        
        Args:
            href: Relative URL path (e.g., '2026/03.02.2026.html')
            
        Returns:
            Dict mapping (region, grade) to average price
        """
        url = f"{BASE_URL}/{href}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        prices = {}
        
        # Find all tables and look for cinnamon prices
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this is the cinnamon table
            header_row = table.find('tr')
            if not header_row:
                continue
            
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            # Look for cinnamon grade columns
            if not any('alba' in h or 'c-5' in h or 'c-4' in h for h in headers):
                continue
            
            # Parse column indices for each grade
            grade_cols = {}
            for idx, header in enumerate(headers):
                for grade_name, grade_code in GRADE_MAPPING.items():
                    if grade_name in header and 'average' in header:
                        grade_cols[grade_code] = idx
                        break
            
            if not grade_cols:
                continue
            
            # Parse rows for each district
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue
                
                district = cells[0].get_text(strip=True).lower()
                district = district.replace('_', ' ').strip()
                
                # Map to our region names
                region = None
                for web_name, our_name in REGION_MAPPING.items():
                    if web_name in district:
                        region = our_name
                        break
                
                if not region:
                    continue
                
                # Extract prices for each grade
                for grade, col_idx in grade_cols.items():
                    if col_idx < len(cells):
                        price_text = cells[col_idx].get_text(strip=True)
                        price = self._parse_price(price_text)
                        if price and price > 0:
                            key = f"{region}_{grade}"
                            prices[key] = price
        
        return prices
    
    def _parse_price(self, text: str) -> Optional[float]:
        """Parse price text to float."""
        if not text or text == '-':
            return None
        # Remove commas and extra characters
        text = re.sub(r'[^\d.]', '', text.replace(',', ''))
        try:
            return float(text)
        except ValueError:
            return None
    
    def get_monthly_average_prices(
        self, 
        year: int, 
        month: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Get monthly average prices by aggregating weekly data.
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            Dict mapping region -> grade -> average price
        """
        weekly_links = self.get_weekly_links_for_month(year, month)
        
        if not weekly_links:
            print(f"No weekly data found for {year}-{month:02d}")
            return {}
        
        print(f"Found {len(weekly_links)} weeks for {year}-{month:02d}")
        
        # Collect all weekly prices
        all_prices = defaultdict(list)
        
        for link in weekly_links:
            prices = self.fetch_weekly_prices(link['href'])
            for key, price in prices.items():
                all_prices[key].append(price)
        
        # Calculate averages
        result = defaultdict(dict)
        for key, price_list in all_prices.items():
            region, grade = key.split('_', 1)  # Split on first _ only (grades like h_faq contain _)
            avg_price = sum(price_list) / len(price_list)
            result[region][grade] = round(avg_price, 2)
        
        return dict(result)
    
    def get_regional_price(
        self, 
        year: int, 
        month: int, 
        region: str, 
        grade: str
    ) -> Optional[float]:
        """
        Get regional price for a specific month/region/grade.
        
        Args:
            year: Year
            month: Month
            region: Region name
            grade: Cinnamon grade
            
        Returns:
            Average price or None
        """
        monthly_prices = self.get_monthly_average_prices(year, month)
        
        region_lower = region.lower()
        grade_lower = grade.lower()
        
        if region_lower in monthly_prices:
            return monthly_prices[region_lower].get(grade_lower)
        
        return None
    
    def get_national_price(
        self,
        year: int,
        month: int,
        grade: str
    ) -> Optional[float]:
        """
        Get national average price (average across all regions).
        
        Args:
            year: Year
            month: Month
            grade: Cinnamon grade
            
        Returns:
            National average price
        """
        monthly_prices = self.get_monthly_average_prices(year, month)
        
        grade_lower = grade.lower()
        prices = []
        
        for region, grades in monthly_prices.items():
            if grade_lower in grades:
                prices.append(grades[grade_lower])
        
        if prices:
            return round(sum(prices) / len(prices), 2)
        
        return None


# Singleton instance
_scraper: Optional[ExagriPriceScraper] = None


def get_exagri_scraper() -> ExagriPriceScraper:
    """Get the singleton cinnamon scraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = ExagriPriceScraper()
    return _scraper


# ============================================================
# PEPPER SCRAPER
# ============================================================

# Pepper grade mapping from website names to dataset names
PEPPER_GRADE_MAPPING = {
    'gr-1': 'gr1',
    'gr-2': 'gr2',
    'white': 'white',
}

# Pepper region mapping from website names to dataset names
PEPPER_REGION_MAPPING = {
    'kandy': 'kandy',
    'matale': 'matale',
    'nuwara_eliya': 'nuwaraeliya',
    'nuwaraeliya': 'nuwaraeliya',
    'kegalle': 'kegalle',
    'ratnapura': 'ratnapura',
    'badulla': 'badulla',
    'kurunegala': 'kurunegala',
    'monaragala': 'monaragala',
    'colombo': 'colombo',
    'galle': 'galle',
    'gampaha': 'gampaha',
    'hambantota': 'hambantota',
    'kalutara': 'kalutara',
    'matara': 'matara',
}


class PepperExagriScraper:
    """Scraper for exagri.info pepper prices (Table 0)."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache_dir = Path(CACHE_DIR) / 'exagri_pepper'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._weekly_links_cache = None
    
    def get_all_weekly_links(self) -> List[Dict]:
        """Fetch all weekly price links from the index page."""
        if self._weekly_links_cache:
            return self._weekly_links_cache
        
        url = f"{BASE_URL}/index.html"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching index: {e}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            match = re.search(r'(\d{4})/(\d{2})\.(\d{2})\.(\d{4})\.html', href)
            if match:
                day = int(match.group(2))
                month = int(match.group(3))
                year = int(match.group(4))
                try:
                    dt = date(year, month, day)
                    links.append({
                        'href': href,
                        'date': dt,
                        'year': year,
                        'month': month,
                        'day': day,
                    })
                except ValueError:
                    continue
        
        links.sort(key=lambda x: x['date'], reverse=True)
        self._weekly_links_cache = links
        return links
    
    def get_weekly_links_for_month(self, year: int, month: int) -> List[Dict]:
        """Get weekly links for a specific month."""
        all_links = self.get_all_weekly_links()
        return [
            link for link in all_links 
            if link['year'] == year and link['month'] == month
        ]
    
    def fetch_weekly_prices(self, href: str) -> Dict[str, float]:
        """
        Fetch pepper prices from a weekly price page.
        Pepper is in Table 0 with columns:
        District, GR-1(Highest), GR-1(Average), GR-2(Highest), GR-2(Average), WHITE(Highest), WHITE(Average)
        """
        url = f"{BASE_URL}/{href}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        prices = {}
        
        tables = soup.find_all('table')
        
        for table in tables:
            header_row = table.find('tr')
            if not header_row:
                continue
            
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            # Look for pepper table: has GR-1 or WHITE columns
            if not any('gr-1' in h or 'gr-2' in h for h in headers):
                continue
            # Exclude cinnamon table (which might have GR references but also has alba)
            if any('alba' in h or 'c-5' in h for h in headers):
                continue
            
            # Parse column indices for each pepper grade (use average price columns)
            grade_cols = {}
            for idx, header in enumerate(headers):
                for grade_name, grade_code in PEPPER_GRADE_MAPPING.items():
                    if grade_name in header and 'average' in header:
                        grade_cols[grade_code] = idx
                        break
            
            if not grade_cols:
                continue
            
            # Parse rows for each district
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue
                
                district = cells[0].get_text(strip=True).lower()
                district = district.replace('_', ' ').strip()
                
                # Map to our region names
                region = None
                for web_name, our_name in PEPPER_REGION_MAPPING.items():
                    if web_name.replace('_', ' ') in district or web_name in district.replace(' ', ''):
                        region = our_name
                        break
                
                if not region:
                    continue
                
                # Extract prices for each grade
                for grade, col_idx in grade_cols.items():
                    if col_idx < len(cells):
                        price_text = cells[col_idx].get_text(strip=True)
                        price = self._parse_price(price_text)
                        if price and price > 0:
                            key = f"{region}_{grade}"
                            prices[key] = price
        
        return prices
    
    def _parse_price(self, text: str) -> Optional[float]:
        """Parse price text to float."""
        if not text or text == '-':
            return None
        text = re.sub(r'[^\d.]', '', text.replace(',', ''))
        try:
            return float(text)
        except ValueError:
            return None
    
    def get_monthly_average_prices(
        self, year: int, month: int
    ) -> Dict[str, Dict[str, float]]:
        """Get monthly average pepper prices by aggregating weekly data."""
        weekly_links = self.get_weekly_links_for_month(year, month)
        
        if not weekly_links:
            print(f"No weekly data found for {year}-{month:02d}")
            return {}
        
        print(f"Found {len(weekly_links)} weeks for {year}-{month:02d}")
        
        all_prices = defaultdict(list)
        
        for link in weekly_links:
            prices = self.fetch_weekly_prices(link['href'])
            for key, price in prices.items():
                all_prices[key].append(price)
        
        result = defaultdict(dict)
        for key, price_list in all_prices.items():
            region, grade = key.split('_', 1)
            avg_price = sum(price_list) / len(price_list)
            result[region][grade] = round(avg_price, 2)
        
        return dict(result)
    
    def get_regional_price(
        self, year: int, month: int, region: str, grade: str
    ) -> Optional[float]:
        """Get regional pepper price for a specific month/region/grade."""
        monthly_prices = self.get_monthly_average_prices(year, month)
        region_lower = region.lower()
        grade_lower = grade.lower()
        
        if region_lower in monthly_prices:
            return monthly_prices[region_lower].get(grade_lower)
        return None


# Pepper singleton
_pepper_scraper: Optional[PepperExagriScraper] = None


def get_pepper_scraper() -> PepperExagriScraper:
    """Get the singleton pepper scraper instance."""
    global _pepper_scraper
    if _pepper_scraper is None:
        _pepper_scraper = PepperExagriScraper()
    return _pepper_scraper


def get_scraper(commodity: str = 'cinnamon'):
    """Factory: get the correct scraper for a commodity."""
    if commodity == 'pepper':
        return get_pepper_scraper()
    return get_exagri_scraper()


def fetch_regional_price_exagri(
    region: str,
    grade: str,
    year: int,
    month: int,
    commodity: str = 'cinnamon'
) -> Optional[float]:
    """Fetch regional price from exagri.info for any commodity."""
    scraper = get_scraper(commodity)
    return scraper.get_regional_price(year, month, region, grade)


def fetch_national_price_exagri(
    grade: str,
    year: int,
    month: int,
    commodity: str = 'cinnamon'
) -> Optional[float]:
    """Fetch national average price from exagri.info."""
    if commodity == 'pepper':
        # Pepper scraper doesn't have national price concept yet
        scraper = get_pepper_scraper()
        monthly_prices = scraper.get_monthly_average_prices(year, month)
        grade_lower = grade.lower()
        prices = []
        for region, grades in monthly_prices.items():
            if grade_lower in grades:
                prices.append(grades[grade_lower])
        if prices:
            return round(sum(prices) / len(prices), 2)
        return None
    else:
        scraper = get_exagri_scraper()
        return scraper.get_national_price(year, month, grade)


if __name__ == "__main__":
    # Test the pepper scraper
    print("=== Pepper Scraper Test ===\n")
    
    scraper = PepperExagriScraper()
    
    print("Fetching monthly average pepper prices for October 2025...")
    prices = scraper.get_monthly_average_prices(2025, 10)
    
    print("\nMonthly Average Pepper Prices:")
    for region, grades in sorted(prices.items()):
        print(f"\n{region.capitalize()}:")
        for grade, price in sorted(grades.items()):
            print(f"  {grade}: Rs. {price}")

