import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath('.'))

try:
    from src import forecasting_engine as engine
    print("Imported engine.")
    
    model = engine.load_artifacts()
    if model:
        print("SUCCESS: Model loaded.")
    else:
        print("FAILURE: Model returned None.")
        
except Exception as e:
    print(f"FAILURE: Exception {e}")
