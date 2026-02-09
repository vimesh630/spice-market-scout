"""
Static data generators for cinnamon dataset.
Handles Grade, Region, Is_Active_Region, and Seasonal_Impact fields.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CINNAMON_GRADES, REGIONS, ACTIVE_REGIONS, SEASONAL_MONTHS
from typing import List, Dict
from datetime import date


def get_grades() -> List[str]:
    """
    Returns the list of cinnamon grades.
    
    Returns:
        List of grade codes: ['alba', 'c4', 'c5', 'c5sp', 'h1', 'h2']
    """
    return CINNAMON_GRADES.copy()


def get_regions() -> List[str]:
    """
    Returns the list of regions for cinnamon data collection.
    
    Returns:
        List of region names
    """
    return REGIONS.copy()


def get_active_regions() -> Dict[str, int]:
    """
    Returns the active region mapping.
    Active regions (1) are major cinnamon producing areas.
    
    Returns:
        Dict mapping region name to active status (0 or 1)
    """
    return ACTIVE_REGIONS.copy()


def is_active_region(region: str) -> int:
    """
    Check if a region is an active cinnamon producing region.
    
    Args:
        region: Region name (lowercase)
        
    Returns:
        1 if active, 0 otherwise
    """
    return ACTIVE_REGIONS.get(region.lower(), 0)


def get_seasonal_impact(month: int) -> int:
    """
    Get seasonal impact value for a given month.
    
    Cinnamon harvest season runs from April to December.
    - Returns 1 for months 4-12 (April to December)
    - Returns 0 for months 1-3 (January to March)
    
    Args:
        month: Month number (1-12)
        
    Returns:
        1 if in season, 0 if off-season
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    return 1 if month in SEASONAL_MONTHS else 0


def get_seasonal_impact_for_date(dt: date) -> int:
    """
    Get seasonal impact for a specific date.
    
    Args:
        dt: Date object
        
    Returns:
        1 if in season, 0 if off-season
    """
    return get_seasonal_impact(dt.month)


def generate_grade_region_combinations() -> List[Dict]:
    """
    Generate all combinations of Grade and Region with their metadata.
    
    Returns:
        List of dicts with keys: grade, region, is_active_region
    """
    combinations = []
    for grade in CINNAMON_GRADES:
        for region in REGIONS:
            combinations.append({
                'Grade': grade,
                'Region': region,
                'Is_Active_Region': is_active_region(region)
            })
    return combinations


if __name__ == "__main__":
    # Test the functions
    print("=== Static Data Generator Test ===\n")
    
    print("Grades:", get_grades())
    print("Regions:", get_regions())
    print("Active Regions:", get_active_regions())
    
    print("\nSeasonal Impact by Month:")
    for m in range(1, 13):
        print(f"  Month {m:2d}: {get_seasonal_impact(m)}")
    
    print(f"\nTotal grade-region combinations: {len(generate_grade_region_combinations())}")
