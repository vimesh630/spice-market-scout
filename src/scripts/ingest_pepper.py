
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, COMMODITY_CONFIG

def ingest_pepper_data():
    """
    Ingest Pepper data from the provided Excel file.
    Drops empty columns and standardizes the format.
    """
    input_file = r'c:\Vimesh\spice-market-scout\notebooks\Pepper_Dataset.xlsx'
    
    # Get config for Pepper
    config = COMMODITY_CONFIG['pepper']
    output_file = os.path.join(PROCESSED_DATA_DIR, config['data_file'])
    
    print(f"Reading data from {input_file}...")
    
    try:
        df = pd.read_excel(input_file)
        
        # Drop columns that are completely empty or known to be empty based on analysis
        cols_to_drop = ['Vietnam_FOB_USD', 'Urea_Price_USD', 'India_Domestic_Price']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Standardize Grade names
        # Map to config casing: GR1, White
        def normalize_grade(g):
            g = str(g).strip()
            if g.lower().replace('-', '') == 'gr1': return 'GR1'
            if g.lower().replace('-', '') == 'gr2': return 'GR2'
            return g.title() # White -> White
            
        df['Grade'] = df['Grade'].apply(normalize_grade)
        
        # Standardize Region names
        # Analysis showed: Region: ['Kandy' 'Matale' 'Nuwaraeliya' ... ]
        # Config expects Title Case
        df['Region'] = df['Region'].str.title()
        
        # Handle 'Regional_Price' formatting/cleanup if strings exist?
        # Analysis showed float64 so it should be fine.
        
        # Ensure 'Is_Active_Region' exists. If not, map from config.
        active_regions = config['active_regions']
        if 'Is_Active_Region' not in df.columns:
            # Config active_regions keys are now Title Case (Badulla)
            # df['Region'] is now Title Case. Match should work.
            df['Is_Active_Region'] = df['Region'].map(lambda x: active_regions.get(x, 0))
            
        # Ensure other required columns for the pipeline/model exist
        # Model expects 'National_Price' sometimes.
        # If missing, we can calculate it as average of regional prices for that date/grade?
        if 'National_Price' not in df.columns:
            print("Calculating National_Price from Regional_Price averages...")
            national_prices = df.groupby(['Date', 'Grade'])['Regional_Price'].transform('mean')
            df['National_Price'] = national_prices.round(2)
            
        # Ensure numeric columns are filled?
        # The pipeline handles filling usually, but safe to have basic filling here if strict.
        # For now, let's keep gaps as is, or maybe fill 0s?
        # The prompt analysis showed 978 non-null prices out of 1349 rows.
        # We should probably drop rows with missing prices or fill them?
        # Let's drop rows where price is missing solely for the initial dataset to be clean?
        # Or keep them and let the pipeline's forward fill handle it later?
        # Strategy: Keep as is, let automated pipeline logic handle missing data.
        
        # Drop rows where Regional_Price is missing
        df = df.dropna(subset=['Regional_Price'])
        
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Successfully saved processed data to {output_file}")
        print(f"Rows: {len(df)}")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing pepper data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ingest_pepper_data()
