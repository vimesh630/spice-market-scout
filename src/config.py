"""
Configuration for data collection pipeline.
Environment variables are used for sensitive data like API keys.
"""
import os
from dataclasses import dataclass
from typing import Dict, List

# === GRADES AND REGIONS ===
CINNAMON_GRADES = ['alba', 'c4', 'c5', 'c5sp', 'h1', 'h2']

REGIONS = ['colombo', 'galle', 'hambantota', 'kandy', 'kurunegala', 'matara']

ACTIVE_REGIONS = {
    'colombo': 1,
    'galle': 1,
    'hambantota': 1,
    'kandy': 0,
    'kurunegala': 0,
    'matara': 1,
}

# === REGIONAL COORDINATES FOR WEATHER DATA ===
# Approximate coordinates for Sri Lankan cinnamon-producing regions
REGION_COORDINATES: Dict[str, tuple] = {
    'colombo': (6.9271, 79.8612),
    'galle': (6.0535, 80.2210),
    'hambantota': (6.1241, 81.1185),
    'kandy': (7.2906, 80.6337),
    'kurunegala': (7.4863, 80.3647),
    'matara': (5.9549, 80.5550),
}

# === SEASONAL MONTHS ===
# April to December are harvest/active season months
SEASONAL_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

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
