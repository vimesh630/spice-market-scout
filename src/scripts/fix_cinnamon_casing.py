import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, CINNAMON_GRADES, REGIONS

def fix_cinnamon_casing():
    file_path = os.path.join(PROCESSED_DATA_DIR, 'cinnamon_prices.csv')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Original Grades:", df['Grade'].unique())
    print("Original Regions:", df['Region'].unique()[:5])
    
    # Grade Mapping
    grade_map = {
        'alba': 'Alba',
        'c5sp': 'C5SP',
        'c5': 'C5',
        'c4': 'C4',
        'h1': 'H1',
        'h2': 'H2',
        'h_faq': 'H FAQ',
        'h faq': 'H FAQ'
    }
    
    # Apply Grade Mapping
    # If grade is already in title/upper case (unlikely but possible), this map needs to handle it or we use a function
    def map_grade(g):
        g_str = str(g)
        return grade_map.get(g_str.lower(), g_str) # Default to original if not in map, but try lower lookup
        
    df['Grade'] = df['Grade'].apply(map_grade)
    
    # Region Mapping (Title Case)
    df['Region'] = df['Region'].str.title()
    
    # Update Is_Active_Region based on new Title Case regions
    from config import ACTIVE_REGIONS
    if 'Is_Active_Region' in df.columns:
         df['Is_Active_Region'] = df['Region'].map(lambda x: ACTIVE_REGIONS.get(x, 0))

    print("Fixed Grades:", df['Grade'].unique())
    print("Fixed Regions:", df['Region'].unique()[:5])
    
    # Save
    try:
        df.to_csv(file_path, index=False)
        print("Saved fixed Cinnamon dataset.")
    except PermissionError:
        print("ERROR: Could not save file. It is locked by another process (likely api.py).")
        print("Please STOP the API server and run this script again.")
        sys.exit(1)

if __name__ == "__main__":
    fix_cinnamon_casing()
