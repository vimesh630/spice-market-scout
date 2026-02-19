import sys
import os
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_engine import train_model, load_and_prepare_data, BASE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_leakage():
    commodity = 'cinnamon'
    data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{commodity}_prices.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    print(f"\n🧪 STARTING LEAKAGE VERIFICATION EXPERIMENT FOR {commodity.upper()} 🧪")
    print("="*60)
    print("Goal: Replicate notebook high scores (R² ~0.95) by enabling random shuffling (Data Leakage).")
    
    # Load Data
    df = load_and_prepare_data(data_path)
    
    # Train with SHUFFLE=TRUE (The 'Notebook Mode')
    # Use 'minmax' scaler as per recent optimization for Cinnamon
    print("\n--- Training Model with SHUFFLE=TRUE ---")
    model, history, results = train_model(
        df, 
        commodity=commodity, 
        epochs=150,  # Match notebook epochs roughly
        scaler_type='minmax', 
        shuffle=True  # <--- THE KEY PARAMETER
    )
    
    print("\n" + "="*60)
    print("🧪 EXPERIMENT RESULTS 🧪")
    print("="*60)
    print(f"R² Score: {results['r2']:.4f}")
    print(f"MAE:      {results['mae']:.4f}")
    print(f"RMSE:     {results['rmse']:.4f}")
    print("-" * 30)
    
    if results['r2'] > 0.90:
        print("✅ HYPOTHESIS CONFIRMED: High R² is due to data leakage (shuffling).")
    else:
        print("❌ HYPOTHESIS REJECTED: R² is still low. Discrepancy is elsewhere.")
    print("="*60)

if __name__ == "__main__":
    verify_leakage()
