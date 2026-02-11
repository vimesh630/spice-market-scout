"""
Data Pipeline Orchestrator for Spice Market Scout.
Coordinates all data collectors to build the complete dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple
import logging

from config import PROCESSED_DATA_DIR, CINNAMON_GRADES, REGIONS, get_commodity_config
from collectors.static_data import (
    get_grades, get_regions, is_active_region, get_seasonal_impact
)
from collectors.weather_collector import fetch_monthly_weather
from collectors.cbsl_collector import fetch_cbsl_data
from collectors.fuel_collector import fetch_fuel_price
from collectors.price_collector import get_price_collector
from collectors.production_collector import get_production_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline orchestrator.
    Collects data from all sources and builds the complete dataset.
    """
    
    def __init__(self, commodity: str = 'cinnamon'):
        """
        Initialize the data pipeline.
        
        Args:
            commodity: Commodity name (default: 'cinnamon')
        """
        self.commodity = commodity
        self.config = get_commodity_config(commodity)
        self.price_collector = get_price_collector(commodity)
        
        # Ensure price cache is populated from existing dataset
        self.price_collector.import_from_existing_dataset(commodity)
    
    def collect_monthly_data(self, year: int, month: int) -> pd.DataFrame:
        """
        Collect all data for a specific month.
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            DataFrame with all data for that month
        """
        logger.info(f"Collecting data for {year}-{month:02d}")
        
        # 1. Get production data (same for all grades/regions)
        production = get_production_data(year, month)
        logger.info(f"  Production data: {production}")
        
        # 2. Get economic indicators
        cbsl_data = fetch_cbsl_data(year, month)
        fuel_price = fetch_fuel_price(year, month)
        logger.info(f"  CBSL data: {cbsl_data}, Fuel: {fuel_price}")
        
        # 3. Get seasonal impact (commodity-aware)
        seasonal_impact = get_seasonal_impact(month, self.commodity)
        
        # 4. Create date
        month_date = date(year, month, 1)
        
        # 5. Collect data for each grade-region combination
        rows = []
        
        for grade in get_grades(self.commodity):
            for region in get_regions(self.commodity):
                # Get weather for this region
                try:
                    weather = fetch_monthly_weather(region, year, month, commodity=self.commodity)
                except Exception as e:
                    logger.warning(f"Weather fetch failed for {region}: {e}")
                    weather = {'temperature': 27.0, 'rainfall': 150.0}
                
                # Get prices (always force refresh to avoid stale cache)
                regional_price = self.price_collector.get_price(
                    year, month, grade, region, 'regional', force_refresh=True
                )
                
                # Build row — common fields across all commodities
                row = {
                    'Date': month_date.strftime('%Y-%m-%d'),
                    'Grade': grade,
                    'Region': region,
                    'Regional_Price': regional_price if regional_price else None,
                    'Seasonal_Impact': seasonal_impact,
                    'Temperature': weather['temperature'],
                    'Rainfall': weather['rainfall'],
                    'Exchange_Rate': cbsl_data['exchange_rate'],
                    'Inflation_Rate': cbsl_data['inflation_rate'],
                    'Fuel_Price': fuel_price,
                }
                
                # Cinnamon-specific fields
                if self.commodity == 'cinnamon':
                    national_price = self.price_collector.get_price(
                        year, month, grade, region, 'national', force_refresh=True
                    )
                    row.update({
                        'Is_Active_Region': is_active_region(region, self.commodity),
                        'National_Price': national_price if national_price else None,
                        'Local_Production_Volume': production['local_production_volume'],
                        'Local_Export_Volume': production['local_export_volume'],
                        'Global_Production_Volume': production['global_production_volume'],
                        'Global_Consumption_Volume': production['global_consumption_volume'],
                        'Market_Sentiment': 'Neutral',
                    })
                
                # Pepper-specific fields
                elif self.commodity == 'pepper':
                    # Vietnam_Harvest_Flag: Feb-May harvest season
                    vietnam_harvest = 1 if month in [2, 3, 4, 5] else 0
                    row.update({
                        'Vietnam_Harvest_Flag': vietnam_harvest,
                    })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"  Collected {len(df)} rows for {year}-{month:02d}")
        return df
    
    def update_dataset(
        self, 
        start_date: date, 
        end_date: date,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Update dataset for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            save: Whether to save to disk
            
        Returns:
            DataFrame with collected data
        """
        logger.info(f"Updating dataset from {start_date} to {end_date}")
        
        all_data = []
        current = start_date.replace(day=1)
        
        while current <= end_date:
            try:
                month_data = self.collect_monthly_data(current.year, current.month)
                all_data.append(month_data)
            except Exception as e:
                logger.error(f"Error collecting data for {current}: {e}")
            
            # Move to next month
            current = current + relativedelta(months=1)
        
        if not all_data:
            logger.warning("No data collected")
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        
        if save:
            self.save_dataset(df, append=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, append: bool = False) -> str:
        """
        Save dataset to disk.
        
        Args:
            df: DataFrame to save
            append: Whether to append to existing dataset
            
        Returns:
            Path to saved file
        """
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        filepath = os.path.join(PROCESSED_DATA_DIR, f'{self.commodity}_prices.csv')
        
        if append and os.path.exists(filepath):
            existing = pd.read_csv(filepath)
            
            # Remove rows that would be duplicated
            existing['Date'] = pd.to_datetime(existing['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Create a key for deduplication
            existing['_key'] = (existing['Date'].astype(str) + '_' + 
                               existing['Grade'] + '_' + 
                               existing['Region'])
            df['_key'] = (df['Date'].astype(str) + '_' + 
                         df['Grade'] + '_' + 
                         df['Region'])
            
            # Remove existing rows that match new data
            existing = existing[~existing['_key'].isin(df['_key'])]
            
            # Drop the key columns
            existing = existing.drop(columns=['_key'])
            df = df.drop(columns=['_key'])
            
            # Combine and sort
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.sort_values(['Region', 'Grade', 'Date'])
            
            # Forward-fill missing/zero prices within each Grade/Region group
            # Use Time-Series Interpolation for better accuracy
            combined['Date'] = pd.to_datetime(combined['Date'])
            combined.set_index('Date', inplace=True)
            
            for col in ['Regional_Price', 'National_Price']:
                if col in combined.columns:
                    # Replace 0 and NaN with forward-filled values
                    combined[col] = combined[col].replace(0, pd.NA)
                    combined[col] = combined.groupby(['Region', 'Grade'])[col].transform(
                        lambda x: x.interpolate(method='time', limit_direction='both')
                    )
                    # Fill any remaining NaN (e.g. if time interpolation fails or single point)
                    combined[col] = combined[col].fillna(method='bfill').fillna(3000.0)

            combined.reset_index(inplace=True)
            
            zero_count = (combined['Regional_Price'] == 0).sum() if 'Regional_Price' in combined.columns else 0
            if zero_count > 0:
                logger.warning(f"  {zero_count} zero prices remain after forward-fill")
            
            combined.to_csv(filepath, index=False)
            logger.info(f"Updated dataset saved to {filepath} ({len(combined)} total rows)")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset saved to {filepath} ({len(df)} rows)")
        
        return filepath
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate collected data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result dict
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
        }
        
        # Check required columns
        required_cols = [
            'Date', 'Grade', 'Region', 'Is_Active_Region', 
            'Regional_Price', 'National_Price', 'Seasonal_Impact',
            'Temperature', 'Rainfall', 'Exchange_Rate', 
            'Inflation_Rate', 'Fuel_Price'
        ]
        
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
        
        # Check for missing prices
        if 'Regional_Price' in df.columns:
            missing_prices = df['Regional_Price'].isna().sum()
            zero_prices = (df['Regional_Price'] == 0).sum()
            if missing_prices > 0:
                results['warnings'].append(f"{missing_prices} missing regional prices")
            if zero_prices > 0:
                results['warnings'].append(f"{zero_prices} zero regional prices")
        
        # Check date range
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            results['date_range'] = {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d'),
            }
        
        return results


def run_pipeline(
    commodity: str = 'cinnamon',
    start_date: str = None,
    end_date: str = None,
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Run the data pipeline.
    
    Args:
        commodity: Commodity name
        start_date: Start date (YYYY-MM or YYYY-MM-DD)
        end_date: End date (YYYY-MM or YYYY-MM-DD)
        dry_run: If True, don't save to disk
        
    Returns:
        Collected DataFrame
    """
    # Parse dates
    if start_date:
        if len(start_date) == 7:  # YYYY-MM format
            start_date += '-01'
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
    else:
        # Default to previous month
        today = date.today()
        start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    
    if end_date:
        if len(end_date) == 7:  # YYYY-MM format
            end_date += '-01'
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
    else:
        end = start
    
    pipeline = DataPipeline(commodity)
    df = pipeline.update_dataset(start, end, save=not dry_run)
    
    # Validate
    validation = pipeline.validate_data(df)
    logger.info(f"Validation: {validation}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data collection pipeline')
    parser.add_argument('--commodity', default='cinnamon', help='Commodity name')
    parser.add_argument('--start', help='Start date (YYYY-MM or YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM or YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help="Don't save to disk")
    
    args = parser.parse_args()
    
    df = run_pipeline(
        commodity=args.commodity,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run
    )
    
    if not df.empty:
        print(f"\nCollected {len(df)} rows:")
        print(df.head(10))
