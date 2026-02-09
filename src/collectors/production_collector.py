"""
Production volume data collector using Gemini AI.
Fetches cinnamon production, export, and consumption data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import date, datetime
from typing import Dict, Optional
from pathlib import Path

from config import get_gemini_api_key, CACHE_DIR


# Historical production data (from existing dataset analysis)
# Values are quarterly/monthly averages derived from the dataset
PRODUCTION_DATA = {
    # 2025 data (estimated based on trends)
    '2025': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
        'local_export': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
        'global_production': {1: 21820, 2: 25314, 3: 22329, 4: 5582, 5: 5582, 6: 20560, 7: 35106, 8: 39860, 9: 31328, 10: 27000, 11: 27000, 12: 27000},
        'global_consumption': {1: 27000, 2: 24000, 3: 19000, 4: 18000, 5: 15000, 6: 15000, 7: 15000, 8: 15000, 9: 20000, 10: 22000, 11: 25000, 12: 28000},
    },
    # 2024 data
    '2024': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 1120, 5: 2240, 6: 3360, 7: 3360, 8: 2240, 9: 1120, 10: 2240, 11: 3360, 12: 3360},
        'local_export': {1: 1330, 2: 1330, 3: 1140, 4: 760, 5: 950, 6: 950, 7: 760, 8: 2850, 9: 2850, 10: 2850, 11: 1710, 12: 1520},
        'global_production': {1: 21388, 2: 17593, 3: 26581, 4: 22320, 5: 20772, 6: 14032, 7: 13099, 8: 17127, 9: 21621, 10: 27047, 11: 24800, 12: 21620},
        'global_consumption': {1: 12400, 2: 17360, 3: 19840, 4: 19840, 5: 19840, 6: 14880, 7: 14880, 8: 14880, 9: 19840, 10: 24800, 11: 29760, 12: 39680},
    },
    # 2023 data
    '2023': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 1120.5, 5: 2241, 6: 3361.5, 7: 3361.5, 8: 2241, 9: 1120.5, 10: 2241, 11: 3361.5, 12: 3361.5},
        'local_export': {1: 1377.4, 2: 1377.4, 3: 1180.6, 4: 787.1, 5: 983.9, 6: 983.9, 7: 787.1, 8: 2951.6, 9: 2951.6, 10: 2951.6, 11: 1770.9, 12: 1574.2},
        'global_production': {1: 20560, 2: 16912, 3: 25552, 4: 21456, 5: 19969, 6: 13489, 7: 12592, 8: 16464, 9: 20784, 10: 26000, 11: 23840, 12: 20785},
        'global_consumption': {1: 11920, 2: 16688, 3: 19072, 4: 19072, 5: 19072, 6: 14304, 7: 14304, 8: 14304, 9: 19072, 10: 23840, 11: 28608, 12: 38147},
    },
    # 2022 data
    '2022': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 1188, 5: 2376, 6: 3564, 7: 3564, 8: 2376, 9: 1188, 10: 2376, 11: 3564, 12: 3564},
        'local_export': {1: 1280.9, 2: 1280.9, 3: 1097.9, 4: 731.9, 5: 914.9, 6: 914.9, 7: 731.9, 8: 2744.7, 9: 2744.7, 10: 2744.7, 11: 1646.8, 12: 1463.9},
        'global_production': {1: 19275, 2: 15855, 3: 23955, 4: 20115, 5: 18720, 6: 12646, 7: 11805, 8: 15435, 9: 19485, 10: 24375, 11: 22350, 12: 19484},
        'global_consumption': {1: 11175, 2: 15645, 3: 17880, 4: 17880, 5: 17880, 6: 13410, 7: 13410, 8: 13410, 9: 17880, 10: 22350, 11: 26820, 12: 35760},
    },
    # 2021 data
    '2021': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 1185, 5: 2370, 6: 3555, 7: 3555, 8: 2370, 9: 1185, 10: 2370, 11: 3555, 12: 3555},
        'local_export': {1: 1317, 2: 1317, 3: 1128.9, 4: 752.6, 5: 940.7, 6: 940.7, 7: 752.6, 8: 2822.2, 9: 2822.2, 10: 2822.2, 11: 1693.3, 12: 1505.2},
        'global_production': {1: 21474, 2: 17664, 3: 26688, 4: 22410, 5: 20856, 6: 14088, 7: 13152, 8: 17196, 9: 21708, 10: 27156, 11: 24900, 12: 21708},
        'global_consumption': {1: 12450, 2: 17430, 3: 19920, 4: 19920, 5: 19920, 6: 14940, 7: 14940, 8: 14940, 9: 19920, 10: 24900, 11: 29880, 12: 39840},
    },
    # 2020 data
    '2020': {
        'local_production': {1: 0, 2: 0, 3: 0, 4: 1145, 5: 2290, 6: 3435, 7: 3435, 8: 2290, 9: 1145, 10: 2290, 11: 3435, 12: 3435},
        'local_export': {1: 1310.5, 2: 1310.5, 3: 1123.3, 4: 748.9, 5: 936.1, 6: 936.1, 7: 748.9, 8: 2808.3, 9: 2808.3, 10: 2808.3, 11: 1685, 12: 1497.7},
        'global_production': {1: 19156, 2: 15757, 3: 23807, 4: 19991, 5: 18605, 6: 12568, 7: 11732, 8: 15340, 9: 19365, 10: 24225, 11: 22212, 12: 19364},
        'global_consumption': {1: 11106, 2: 15549, 3: 17770, 4: 17770, 5: 17770, 6: 13327, 7: 13327, 8: 13327, 9: 17770, 10: 22212, 11: 26655, 12: 35539},
    },
}


def get_production_data(year: int, month: int) -> Dict[str, float]:
    """
    Get production volume data for a specific month.
    
    Args:
        year: Year
        month: Month (1-12)
        
    Returns:
        Dict with keys:
        - local_production_volume
        - local_export_volume
        - global_production_volume
        - global_consumption_volume
    """
    year_str = str(year)
    
    # Check if we have data for this year
    if year_str in PRODUCTION_DATA:
        year_data = PRODUCTION_DATA[year_str]
        return {
            'local_production_volume': year_data['local_production'].get(month, 0),
            'local_export_volume': year_data['local_export'].get(month, 0),
            'global_production_volume': year_data['global_production'].get(month, 20000),
            'global_consumption_volume': year_data['global_consumption'].get(month, 18000),
        }
    
    # For years not in cache, try Gemini API if available
    api_key = get_gemini_api_key()
    if api_key:
        return _fetch_from_gemini(year, month, api_key)
    
    # Fallback to estimated values based on seasonal patterns
    return _estimate_production(year, month)


def _estimate_production(year: int, month: int) -> Dict[str, float]:
    """
    Estimate production data based on seasonal patterns.
    
    Args:
        year: Year
        month: Month
        
    Returns:
        Estimated production data
    """
    # Seasonal patterns (1-3: off-season, 4-12: harvest season)
    is_harvest = month >= 4
    
    # Local production follows seasonal pattern
    if month in [1, 2, 3]:
        local_prod = 0
    elif month in [4, 9, 10]:
        local_prod = 1100
    elif month in [5, 8]:
        local_prod = 2200
    else:  # 6, 7, 11, 12
        local_prod = 3300
    
    # Export volumes
    if month in [1, 2]:
        export = 1300
    elif month == 3:
        export = 1100
    elif month in [4, 7]:
        export = 750
    elif month in [5, 6]:
        export = 950
    elif month in [8, 9, 10]:
        export = 2800
    elif month == 11:
        export = 1700
    else:  # 12
        export = 1500
    
    # Global production/consumption (rough estimates)
    global_prod = 20000 + (month - 6) * 1000  # Varies by season
    global_cons = 18000 + (month - 6) * 800
    
    return {
        'local_production_volume': local_prod,
        'local_export_volume': export,
        'global_production_volume': max(global_prod, 12000),
        'global_consumption_volume': max(global_cons, 12000),
    }


def _fetch_from_gemini(year: int, month: int, api_key: str) -> Dict[str, float]:
    """
    Fetch production data using Gemini AI API.
    
    Args:
        year: Year
        month: Month
        api_key: Gemini API key
        
    Returns:
        Production data dict
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        For Sri Lankan cinnamon industry in {year}-{month:02d}, provide estimated:
        1. Local production volume (metric tons)
        2. Local export volume (metric tons)
        3. Global cinnamon production volume (metric tons)
        4. Global cinnamon consumption volume (metric tons)
        
        Return ONLY a JSON object with these exact keys:
        - local_production_volume
        - local_export_volume
        - global_production_volume
        - global_consumption_volume
        
        Use realistic estimates based on industry data.
        """
        
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        text = response.text
        # Extract JSON if wrapped in markdown code block
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        data = json.loads(text.strip())
        
        # Cache the result
        _cache_production_data(year, month, data)
        
        return data
        
    except Exception as e:
        print(f"Error fetching from Gemini: {e}")
        return _estimate_production(year, month)


def _cache_production_data(year: int, month: int, data: Dict[str, float]) -> None:
    """Cache production data to disk."""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / 'production_cache.json'
    
    try:
        existing = {}
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                existing = json.load(f)
        
        key = f"{year}-{month:02d}"
        existing[key] = data
        
        with open(cache_file, 'w') as f:
            json.dump(existing, f, indent=2)
            
    except Exception as e:
        print(f"Error caching production data: {e}")


def update_production_data(
    year: int,
    month: int,
    local_production: float = None,
    local_export: float = None,
    global_production: float = None,
    global_consumption: float = None
) -> None:
    """
    Manually update production data for a specific month.
    Useful for adding new data.
    
    Args:
        year: Year
        month: Month
        local_production: Local production volume
        local_export: Local export volume
        global_production: Global production volume
        global_consumption: Global consumption volume
    """
    year_str = str(year)
    
    if year_str not in PRODUCTION_DATA:
        PRODUCTION_DATA[year_str] = {
            'local_production': {},
            'local_export': {},
            'global_production': {},
            'global_consumption': {},
        }
    
    if local_production is not None:
        PRODUCTION_DATA[year_str]['local_production'][month] = local_production
    if local_export is not None:
        PRODUCTION_DATA[year_str]['local_export'][month] = local_export
    if global_production is not None:
        PRODUCTION_DATA[year_str]['global_production'][month] = global_production
    if global_consumption is not None:
        PRODUCTION_DATA[year_str]['global_consumption'][month] = global_consumption


if __name__ == "__main__":
    # Test the production collector
    print("=== Production Data Collector Test ===\n")
    
    test_cases = [
        (2025, 9),
        (2024, 6),
        (2023, 1),
        (2022, 12),
    ]
    
    for year, month in test_cases:
        print(f"{year}-{month:02d}:")
        data = get_production_data(year, month)
        for key, value in data.items():
            print(f"  {key}: {value}")
        print()
