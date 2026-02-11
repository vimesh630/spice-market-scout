
import pandas as pd
import os
import sys
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, COMMODITY_CONFIG
from collectors.exagri_scraper import get_clove_scraper
from collectors.external_data import fetch_external_data_row
from forecasting_engine import train_all_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_clove_data():
    """
    Main function to update Clove data.
    1. Check existing dataset (clove_prices.csv).
    2. Determine missing months (start from last date or create new).
    3. Fetch Prices from Exagri.
    4. Fetch External Data (Weather, Economy, Market).
    5. Merge and Save.
    6. Retrain Model.
    """
    logger.info("Starting Clove Data Update...")
    
    file_path = os.path.join(PROCESSED_DATA_DIR, 'clove_prices.csv')
    
    # Load existing or create new
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        start_date = last_date + relativedelta(months=1)
        is_new = False
    else:
        # Start from a reasonable date if no data exists, e.g., 2024-01-01 or earlier if we want history
        # Let's start from Jan 2024 to get some history if possible, or June 2024 like Pepper
        start_date = datetime(2024, 1, 1) 
        df = pd.DataFrame()
        is_new = True
    
    end_date = datetime.now().replace(day=1)
    
    if start_date > end_date:
        logger.info("Data is already up to date. Proceeding to training check.")
        # We still want to check if new_rows exists for logic below, 
        # but since we skipped fetching, new_rows is empty.
        new_rows = []
    else:
        logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}...")
        
        scraper = get_clove_scraper()
        new_rows = []
        
        current = start_date
        while current <= end_date:
            logger.info(f"  Fetching {current.strftime('%Y-%m')}...")
            
            # 1. Fetch Prices
            try:
                prices = scraper.get_monthly_average_prices(current.year, current.month)
                
                if prices:
                    # Iterate regions found in prices
                    for region, grades in prices.items():
                        for grade, price in grades.items():
                            
                            # 2. Fetch External Data for this region/month
                            # This includes Weather, Economy, Market + Seasonal Impact
                            external_data = fetch_external_data_row(region, current.year, current.month)
                            
                            row = {
                                'Date': current,
                                'Year': current.year,
                                'Month': current.month, # Number
                                'Region': region.title(), # Enforce Title Case
                                'Grade': grade.title(),   # Enforce Title Case
                                'Regional_Price': price,
                                'National_Price': price, # Placeholder/Proxy
                            }
                            
                            # Merge external data
                            row.update(external_data)
                            
                            new_rows.append(row)
                else:
                    logger.warning(f"    No Clove price data found for {current.strftime('%Y-%m')}")
                    
            except Exception as e:
                logger.error(f"    Error fetching: {e}")
            
            current += relativedelta(months=1)
            time.sleep(1) # Polite delay

    if not new_rows:
        logger.info("No new data fetched.")
        if is_new:
            logger.warning("Created empty dataset (no history found).")
            # If new and empty, we can't train.
            return
    else:
        new_df = pd.DataFrame(new_rows)
        
        # Calculate true national price per month/grade if possible
        national_avgs = new_df.groupby(['Date', 'Grade'])['Regional_Price'].transform('mean')
        new_df['National_Price'] = national_avgs.round(2)
        
        if is_new:
            final_df = new_df
        else:
            # Align columns: just concat, let pandas handle missing/extra
            final_df = pd.concat([df, new_df], ignore_index=True)
            
        # Sort
        if 'Date' in final_df.columns:
            final_df = final_df.sort_values(['Date', 'Region', 'Grade'])
            
        # Save
        final_df.to_csv(file_path, index=False)
        logger.info(f"Updated dataset saved to {file_path}")
        logger.info(f"New total rows: {len(final_df)}")

    # 3. Retrain Model
    logger.info("Step 3: Retraining Clove Model...")
    try:
        # We need to reload the data from file to ensure clean state
        # train_all_models handles loading and training
        train_all_models(commodity='clove')
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")


if __name__ == "__main__":
    update_clove_data()
