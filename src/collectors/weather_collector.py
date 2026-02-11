"""
Weather data collector using Open-Meteo Historical Weather API.
Fetches temperature and rainfall data for Sri Lankan cinnamon regions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
import json

from config import REGION_COORDINATES, OPEN_METEO_HISTORICAL_URL


def fetch_weather_data(
    region: str,
    start_date: str,
    end_date: str
) -> Dict[str, float]:
    """
    Fetch historical weather data for a region using Open-Meteo API.
    
    Args:
        region: Region name (e.g., 'colombo', 'galle')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dict with 'temperature' (°C) and 'rainfall' (mm) monthly averages
        
    Raises:
        ValueError: If region is not found in coordinates
        requests.RequestException: If API request fails
    """

    # Use Title Case for lookup as per new config standard
    region_title = region.title()
    if region_title not in REGION_COORDINATES:
        raise ValueError(f"Unknown region: {region}. Available: {list(REGION_COORDINATES.keys())}")
    
    lat, lon = REGION_COORDINATES[region_title]
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_mean,precipitation_sum',
        'timezone': 'Asia/Colombo'
    }
    
    try:
        response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get('daily', {})
        temperatures = daily.get('temperature_2m_mean', [])
        precipitations = daily.get('precipitation_sum', [])
        
        # Filter out None values and calculate averages
        valid_temps = [t for t in temperatures if t is not None]
        valid_precips = [p for p in precipitations if p is not None]
        
        avg_temp = sum(valid_temps) / len(valid_temps) if valid_temps else 0.0
        total_rainfall = sum(valid_precips) if valid_precips else 0.0
        
        return {
            'temperature': round(avg_temp, 2),
            'rainfall': round(total_rainfall, 2)
        }
        
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        raise


def fetch_monthly_weather(region: str, year: int, month: int) -> Dict[str, float]:
    """
    Fetch weather data for a specific month.
    
    Args:
        region: Region name
        year: Year (e.g., 2025)
        month: Month number (1-12)
        
    Returns:
        Dict with 'temperature' and 'rainfall'
    """
    # Calculate first and last day of month
    first_day = date(year, month, 1)
    
    # Get last day of month
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    
    # Don't fetch future dates
    today = date.today()
    if last_day > today:
        last_day = today - timedelta(days=1)
    
    if first_day > today:
        # Return defaults for future months
        return {'temperature': 27.0, 'rainfall': 150.0}
    
    return fetch_weather_data(
        region,
        first_day.strftime('%Y-%m-%d'),
        last_day.strftime('%Y-%m-%d')
    )


def fetch_weather_for_all_regions(year: int, month: int) -> Dict[str, Dict[str, float]]:
    """
    Fetch weather data for all regions for a specific month.
    
    Args:
        year: Year
        month: Month number
        
    Returns:
        Dict mapping region name to weather data
    """
    results = {}
    for region in REGION_COORDINATES.keys():
        try:
            results[region] = fetch_monthly_weather(region, year, month)
        except Exception as e:
            print(f"Error fetching weather for {region}: {e}")
            # Use default values on error
            results[region] = {'temperature': 27.0, 'rainfall': 150.0}
    return results


if __name__ == "__main__":
    # Test the weather collector
    print("=== Weather Collector Test ===\n")
    
    # Test for a specific month
    test_year = 2025
    test_month = 9
    
    print(f"Fetching weather data for {test_year}-{test_month:02d}...\n")
    
    for region in ['colombo', 'galle', 'matara']:
        try:
            weather = fetch_monthly_weather(region, test_year, test_month)
            print(f"{region.capitalize()}:")
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Rainfall: {weather['rainfall']} mm")
        except Exception as e:
            print(f"{region}: Error - {e}")
