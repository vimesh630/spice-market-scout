import os
import pandas as pd
from src import forecasting_engine as engine

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def rebuild():
    commodities = ['cinnamon', 'pepper', 'clove']
    
    print("\n--- REBUILDING ALL MODELS ---")
    
    for com in commodities:
        print(f"\nProcessing {com}...")
        file_path = os.path.join(DATA_DIR, f'{com}_prices.csv')
        
        if not os.path.exists(file_path):
            print(f"  [SKIPPED] Data file not found: {file_path}")
            continue
            
        try:
            print(f"  Loading data...")
            df = engine.load_and_prepare_data(file_path)
            
            if len(df) < 50:
                print(f"  [SKIPPED] Insufficient data ({len(df)} rows)")
                continue
                
            print(f"  Training model (30 epochs)...")
            engine.train_model(df, commodity=com, epochs=30)
            print(f"  [SUCCESS] {com} model rebuilt.")
            
        except Exception as e:
            print(f"  [ERROR] Failed to rebuild {com}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    rebuild()
