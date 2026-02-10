
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_engine import train_model, load_and_prepare_data
from config import COMMODITY_CONFIG, PROCESSED_DATA_DIR

def train_pepper_model():
    """
    Train the LSTM model for Pepper.
    """
    commodity = 'pepper'
    config = COMMODITY_CONFIG[commodity]
    data_file = os.path.join(PROCESSED_DATA_DIR, config['data_file'])
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please run ingest_pepper.py first.")
        return
        
    print(f"Loading data for {commodity} from {data_file}...")
    df = load_and_prepare_data(data_file)
    
    print(f"Training model for {commodity}...")
    # Using default tuning settings (Optuna)
    # Reducing epochs/trials for quick feedback loop in this environment
    train_model(
        df, 
        commodity=commodity, 
        use_tuning=True, 
        tuning_method='random',  # Random is often faster/good enough for initial
        n_tuning_trials=5,      # Small number for speed
        epochs=30
    )
    
    print("Training complete.")

if __name__ == "__main__":
    train_pepper_model()
