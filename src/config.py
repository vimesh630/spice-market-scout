"""
Configuration for data collection pipeline.
Environment variables are used for sensitive data like API keys.
"""
import os
from dataclasses import dataclass
from typing import Dict, List

# === CINNAMON GRADES AND REGIONS ===
CINNAMON_GRADES = ['alba', 'c4', 'c5', 'c5sp', 'h1', 'h2', 'h_faq']

REGIONS = ['colombo', 'galle', 'hambantota', 'kalutara', 'kandy', 'kurunegala', 'matara', 'ratnapura']

ACTIVE_REGIONS = {
    'colombo': 1,
    'galle': 1,
    'hambantota': 1,
    'kandy': 0,
    'kurunegala': 0,
    'matara': 1,
    'ratnapura': 1,
    'kalutara': 1,
}

# Approximate coordinates for Sri Lankan cinnamon-producing regions
REGION_COORDINATES: Dict[str, tuple] = {
    'colombo': (6.9271, 79.8612),
    'galle': (6.0535, 80.2210),
    'hambantota': (6.1241, 81.1185),
    'kandy': (7.2906, 80.6337),
    'kurunegala': (7.4863, 80.3647),
    'matara': (5.9549, 80.5550),
    'ratnapura': (6.6828, 80.3992),
    'kalutara': (6.5854, 79.9607),
}

# === PEPPER GRADES AND REGIONS ===
PEPPER_GRADES = ['gr1', 'white']

PEPPER_REGIONS = [
    'badulla', 'colombo', 'galle', 'gampaha', 'hambantota', 'kalutara',
    'kandy', 'kegalle', 'kurunegala', 'matale', 'matara', 'monaragala',
    'nuwaraeliya', 'ratnapura'
]

PEPPER_ACTIVE_REGIONS = {r: 1 for r in PEPPER_REGIONS}

PEPPER_REGION_COORDINATES: Dict[str, tuple] = {
    'badulla': (6.9934, 81.0550),
    'colombo': (6.9271, 79.8612),
    'galle': (6.0535, 80.2210),
    'gampaha': (7.0840, 80.0098),
    'hambantota': (6.1241, 81.1185),
    'kalutara': (6.5854, 79.9607),
    'kandy': (7.2906, 80.6337),
    'kegalle': (7.2513, 80.3464),
    'kurunegala': (7.4863, 80.3647),
    'matale': (7.4675, 80.6234),
    'matara': (5.9549, 80.5550),
    'monaragala': (6.8728, 81.3507),
    'nuwaraeliya': (6.9497, 80.7891),
    'ratnapura': (6.6828, 80.3992),
}

# === SEASONAL MONTHS ===
# April to December are harvest/active season months (cinnamon)
SEASONAL_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
# Pepper harvest season: typically Feb-May main crop
PEPPER_SEASONAL_MONTHS = [2, 3, 4, 5]

# === COMMODITY CONFIGURATION REGISTRY ===
COMMODITY_CONFIG = {
    'cinnamon': {
        'grades': CINNAMON_GRADES,
        'regions': REGIONS,
        'active_regions': ACTIVE_REGIONS,
        'coordinates': REGION_COORDINATES,
        'seasonal_months': SEASONAL_MONTHS,
        'data_file': 'cinnamon_prices.csv',
        'exagri_table': 'cinnamon',
    },
    'pepper': {
        'grades': PEPPER_GRADES,
        'regions': PEPPER_REGIONS,
        'active_regions': PEPPER_ACTIVE_REGIONS,
        'coordinates': PEPPER_REGION_COORDINATES,
        'seasonal_months': PEPPER_SEASONAL_MONTHS,
        'data_file': 'pepper_prices.csv',
        'exagri_table': 'pepper',
    },
}

def get_commodity_config(commodity: str) -> dict:
    """Get configuration for a specific commodity."""
    if commodity not in COMMODITY_CONFIG:
        raise ValueError(f"Unknown commodity: {commodity}. Available: {list(COMMODITY_CONFIG.keys())}")
    return COMMODITY_CONFIG[commodity]

# === API ENDPOINTS ===
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

CBSL_EXCHANGE_RATE_URL = "https://www.cbsl.gov.lk/rates-and-indicators/exchange-rates"
CBSL_INFLATION_URL = "https://www.cbsl.gov.lk/en/measures-of-consumer-price-inflation"

CEYPETCO_URL = "https://ceypetco.gov.lk"

EXAGRI_URL = "https://exagri.info/mkt/index.html"

# === API KEYS (from environment) ===
def get_gemini_api_key() -> str:
    """Get Gemini API key from environment variable."""
    key = os.environ.get('GEMINI_API_KEY', '')
    if not key:
        key = os.environ.get('GOOGLE_API_KEY', '')
    return key

# === DATA PATHS ===
# Get the project root directory (parent of src)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# === CACHE SETTINGS ===
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
CACHE_EXPIRY_DAYS = 7  # Production data cache expiry

