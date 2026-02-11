"""
Static data generators for spice datasets.
Handles Grade, Region, Is_Active_Region, and Seasonal_Impact fields.
Supports multiple commodities via COMMODITY_CONFIG.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CINNAMON_GRADES, REGIONS, ACTIVE_REGIONS, SEASONAL_MONTHS,
    get_commodity_config
)
from typing import List, Dict
from datetime import date


def get_grades(commodity: str = 'cinnamon') -> List[str]:
    """
    Returns the list of grades for a commodity.
    """
    config = get_commodity_config(commodity)
    return config['grades'].copy()


def get_regions(commodity: str = 'cinnamon') -> List[str]:
    """
    Returns the list of regions for a commodity.
    """
    config = get_commodity_config(commodity)
    return config['regions'].copy()


def get_active_regions(commodity: str = 'cinnamon') -> Dict[str, int]:
    """
    Returns the active region mapping for a commodity.
    """
    config = get_commodity_config(commodity)
    return config['active_regions'].copy()


def is_active_region(region: str, commodity: str = 'cinnamon') -> int:
    """
    Check if a region is an active producing region for a commodity.
    """
    config = get_commodity_config(commodity)
    # config keys are now Title Case for all commodities
    return config['active_regions'].get(region.title(), 0)


def get_seasonal_impact(month: int, commodity: str = 'cinnamon') -> int:
    """
    Get seasonal impact value for a given month and commodity.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    config = get_commodity_config(commodity)
    return 1 if month in config['seasonal_months'] else 0


def get_seasonal_impact_for_date(dt: date, commodity: str = 'cinnamon') -> int:
    """
    Get seasonal impact for a specific date and commodity.
    """
    return get_seasonal_impact(dt.month, commodity)


def generate_grade_region_combinations(commodity: str = 'cinnamon') -> List[Dict]:
    """
    Generate all combinations of Grade and Region with their metadata.
    """
    config = get_commodity_config(commodity)
    combinations = []
    for grade in config['grades']:
        for region in config['regions']:
            combinations.append({
                'Grade': grade,
                'Region': region,
                'Is_Active_Region': config['active_regions'].get(region, 0)
            })
    return combinations


if __name__ == "__main__":
    for com in ['cinnamon', 'pepper']:
        print(f"\n=== {com.upper()} Static Data ===")
        print("Grades:", get_grades(com))
        print("Regions:", get_regions(com))
        print(f"Total combinations: {len(generate_grade_region_combinations(com))}")

