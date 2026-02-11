"""
Configuration for data collection pipeline.
Environment variables are used for sensitive data like API keys.
"""
import os
from dataclasses import dataclass
from typing import Dict, List

# === CINNAMON GRADES AND REGIONS ===
CINNAMON_GRADES = ['Alba', 'C5SP', 'C5', 'C4', 'H1', 'H2', 'H FAQ']

REGIONS = ['Colombo', 'Galle', 'Hambantota', 'Kalutara', 'Kandy', 'Kurunegala', 'Matara', 'Ratnapura']

ACTIVE_REGIONS = {
    'Colombo': 1,
    'Galle': 1,
    'Hambantota': 1,
    'Kandy': 0,
    'Kurunegala': 0,
    'Matara': 1,
    'Ratnapura': 1,
    'Kalutara': 1,
}

# Approximate coordinates for Sri Lankan cinnamon-producing regions
REGION_COORDINATES: Dict[str, tuple] = {
    'Colombo': (6.9271, 79.8612),
    'Galle': (6.0535, 80.2210),
    'Hambantota': (6.1241, 81.1185),
    'Kandy': (7.2906, 80.6337),
    'Kurunegala': (7.4863, 80.3647),
    'Matara': (5.9549, 80.5550),
    'Ratnapura': (6.6828, 80.3992),
    'Kalutara': (6.5854, 79.9607),
}

# === PEPPER GRADES AND REGIONS ===
# === PEPPER GRADES AND REGIONS ===
PEPPER_GRADES = ['GR1', 'White']

PEPPER_REGIONS = [
    'Badulla', 'Colombo', 'Galle', 'Gampaha', 'Hambantota', 'Kalutara',
    'Kandy', 'Kegalle', 'Kurunegala', 'Matale', 'Matara', 'Monaragala',
    'Nuwaraeliya', 'Ratnapura'
]

PEPPER_ACTIVE_REGIONS = {r: 1 for r in PEPPER_REGIONS}

PEPPER_REGION_COORDINATES: Dict[str, tuple] = {
    'Badulla': (6.9934, 81.0550),
    'Colombo': (6.9271, 79.8612),
    'Galle': (6.0535, 80.2210),
    'Gampaha': (7.0840, 80.0098),
    'Hambantota': (6.1241, 81.1185),
    'Kalutara': (6.5854, 79.9607),
    'Kandy': (7.2906, 80.6337),
    'Kegalle': (7.2513, 80.3464),
    'Kurunegala': (7.4863, 80.3647),
    'Matale': (7.4675, 80.6234),
    'Matara': (5.9549, 80.5550),
    'Monaragala': (6.8728, 81.3507),
    'Nuwaraeliya': (6.9497, 80.7891),
    'Ratnapura': (6.6828, 80.3992),
}

# === SEASONAL MONTHS ===
# April to December are harvest/active season months (cinnamon)
# Maintain lowercase for Cinnamon as per dataset check
SEASONAL_MONTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12]
# Pepper harvest season: typically Feb-May main crop
PEPPER_SEASONAL_MONTHS = [2, 3, 4, 5]

# === CLOVE GRADES AND REGIONS ===
CLOVE_GRADES = ['Clove', 'Stem']

# Clove regions (assuming similar to Pepper/Cinnamon based on "Same regions are repeating")
# using Pepper regions as they are extensive
CLOVE_REGIONS = PEPPER_REGIONS
CLOVE_ACTIVE_REGIONS = PEPPER_ACTIVE_REGIONS
CLOVE_REGION_COORDINATES = PEPPER_REGION_COORDINATES

# Clove harvest season: Typically Dec-April
CLOVE_SEASONAL_MONTHS = [12, 1, 2, 3, 4]

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
    'clove': {
        'grades': CLOVE_GRADES,
        'regions': CLOVE_REGIONS,
        'active_regions': CLOVE_ACTIVE_REGIONS,
        'coordinates': CLOVE_REGION_COORDINATES,
        'seasonal_months': CLOVE_SEASONAL_MONTHS,
        'data_file': 'clove_prices.csv',
        'exagri_table': 'clove',
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

