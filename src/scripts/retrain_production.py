import sys
import os
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_engine import train_model, load_and_prepare_data, BASE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_production_models():
    print("\n🏭 STARTING PRODUCTION RETRAINING (CHRONOLOGICAL SPLIT) 🏭")
    print("="*60)
    
    commodities = ['cinnamon', 'pepper']
    
    for commodity in commodities:
        print(f"\nTraining {commodity.upper()} Model...")
        data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{commodity}_prices.csv')
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            continue

        df = load_and_prepare_data(data_path)
        
        # Use 'minmax' for target scaling as it tends to be more stable for prices
        # Shuffle=False is default, but explicit here for clarity
        model, history, results = train_model(
            df, 
            commodity=commodity, 
            epochs=200, 
            scaler_type='standard', # standard scaler might be better for gradients, let's stick to standard for consistency with notebooks features?
            # Wait, verify_leakage used 'minmax'. Notebook used 'minmax' for target? 
            # Notebook cell 153 says "y_test_unscaled = scaler_target.inverse_transform...".
            # Checking recent task.md change 13: "Change 6: Use StandardScaler for target instead of MinMaxScaler"
            # BUT verify_leakage.py used 'minmax'.
            # Let's stick to 'standard' for production if that was the plan, OR 'minmax' if that gave 0.90.
            # R2=0.90 with minmax. 
            # Let's try 'standard' for production to be robust to outliers? Or 'minmax' to match range.
            # I will use 'minmax' since it worked well in verification.
            shuffle=False 
        )
        
        print(f"✅ {commodity.upper()} Trained.")
        print(f"  R²: {results['r2']:.4f}")
        print(f"  MAE: {results['mae']:.2f}")
        print(f"  DA: {results['directional_accuracy']:.2%}")

if __name__ == "__main__":
    train_production_models()
