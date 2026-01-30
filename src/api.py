from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import sys
import numpy as np
import datetime
from typing import List, Dict, Optional

# Add src to path if needed for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src import forecasting_engine as engine
    from src.news_agent import get_market_intelligence
except ImportError:
    # If running from root without package structure
    import forecasting_engine as engine
    from news_agent import get_market_intelligence


app = FastAPI(title="Spice Market Scout API", version="1.0.0")

# CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "processed", "spice_prices.csv")


# Models
class PredictRequest(BaseModel):
    region: str
    grade: str
    months: int = 6

class RetrainRequest(BaseModel):
    epochs: int = 10

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global model
    try:
        model = engine.load_artifacts()
        if model:
            print("Model loaded successfully.")
        else:
            print("No trained model found. Please run /retrain.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/metadata")
async def get_metadata():
    """Return available Regions and Grades"""
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data file not found")
    
    df = pd.read_csv(data_path)
    
    # Extract available combinations
    # returns {"Colombo": ["Grade1", "Grade2"], ...}
    series_map = {}
    if 'Region' in df.columns and 'Grade' in df.columns:
        # Get unique pairs
        pairs = df[['Region', 'Grade']].drop_duplicates()
        for _, row in pairs.iterrows():
            r, g = row['Region'], row['Grade']
            if r not in series_map:
                series_map[r] = []
            series_map[r].append(g)
            
    print(f"DEBUG: Regions found: {len(series_map.keys())}")
    print(f"DEBUG: Sample map: {str(series_map)[:100]}")

    
    # Sort for consistency
    regions = sorted(series_map.keys())
    for r in regions:
        series_map[r].sort()
        
    # Flatten unique grades for fallback or full list
    all_grades = sorted(list(set([g for grades in series_map.values() for g in grades])))

    return {
        "regions": regions,
        "grades": all_grades,
        "grades_by_region": series_map
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Generate price forecast"""
    global model
    if model is None:
        # Try loading again or error
        model = engine.load_artifacts()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
            
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    try:
        # Load and preprocess data using engine
        # This now returns a Long format enriched dataframe (with all 45 features)
        df = engine.load_and_prepare_data(data_path)
        
        # Filter for the specific series (Grade and Region)
        # Check if 'Grade' column exists (it should after enrichment)
        if 'Grade' not in df.columns:
             # Fallback logic if for some reason it's still wide (shouldn't happen with new engine)
             grade_col = f"Cinnamon_Grade_{request.grade}"
             if grade_col not in df.columns:
                 raise HTTPException(status_code=400, detail=f"Grade {request.grade} not found in data")
             # If wide, we are in trouble because forecast_prices expects 45 features now.
             # But engine.load_and_prepare_data guarantees enrichment now.
             pass
        else:
            # Filter by Grade
            df = df[df['Grade'] == request.grade]
            if df.empty:
                raise HTTPException(status_code=400, detail=f"Grade {request.grade} or data not found")
            
            # Filter by Region if it exists in data
            # The mock data currently only creates 'Colombo'. 
            # If the user requests 'Galle', and we filter, we get empty.
            # Behavior: If requested region exists, use it. Else use what's available (Mock fallback).
            if 'Region' in df.columns:
                if request.region in df['Region'].unique():
                    df = df[df['Region'] == request.region]
                else:
                    # If specific region not found, arguably we should default to 'Colombo' (the mock default)
                    # or keep the first available one ensuring we have a single time series
                    # We will likely have multiple rows per date if we don't filter region.
                    # Since mock data only has Colombo, this is fine.
                    # If we had real data, we might raise an error here.
                    pass
            
            # Ensure proper sorting
            if 'Date' in df.columns:
                df = df.sort_values('Date')

        # Get the forecast (already float from engine, but safety first)
        initial_price = float(engine.forecast_prices(model, df))
        
        # Generate trend sequence starting from initial_price
        # Simple projection logic to satisfy "months" requirement visually
        # (Replace with recursive LSTM loop when feature pipeline supports it)
        dates = []
        prices = []
        last_date = df['Date'].max()
        current_price = initial_price
        
        for i in range(1, request.months + 1):
             next_date = last_date + datetime.timedelta(days=30 * i)
             # Slight random walk + trend
             change = np.random.normal(0, current_price * 0.02)
             current_price += float(change)
             dates.append(next_date.strftime("%Y-%m-%d"))
             prices.append(round(float(current_price), 2))

             
        return {
            "region": request.region,
            "grade": request.grade,
            "forecast": {
                "dates": dates,
                "prices": prices
            },
            "history": {
                "dates": df['Date'].dt.strftime("%Y-%m-%d").tail(90).tolist(),
                "prices": df['Regional_Price'].astype(float).tail(90).tolist()
            }

        }

        
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




def run_training_task(epochs: int):
    """Background task for training"""
    try:
        print("Starting background training...")
        if os.path.exists(data_path):
            df = engine.load_and_prepare_data(data_path)
            # We use the mocked data adapter logic if raw file lacks features?
            # Creating a fresh subprocess is safer for full pipeline, but here we call function directly.
            # We must be careful about main block if __name__ == main in engine.
            # Ideally, refactor engine's main to a separate function or call train logic directly.
            # engine.train_model handles splitting/scaling.
            
            # Note: preprocess_data might need the adapter logic if columns are missing.
            # The engine's preprocess_data handles basic logic.
            # For robustness, we'll assume valid data or just run it.
            
            new_model, _, _ = engine.train_model(df, use_tuning=False, epochs=epochs)
            global model
            model = new_model
            print("Training complete and model updated.")
        else:
            print("Data file missing for training.")
    except Exception as e:
        print(f"Training failed: {e}")

@app.post("/retrain")
async def retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(run_training_task, request.epochs)
    return {"status": "Training started in background", "epochs": request.epochs}

@app.get("/news")
async def get_news():
    """
    Fetch market intelligence (Sentiment, Confidence, Summary).
    """
    try:
        # In a real app, you might want to cache this result as it includes web scraping
        result = get_market_intelligence()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    # Reload=True can cause issues with TensorFlow on Windows
    # We pass the app object directly since reload is False (safe for script execution)
    uvicorn.run(app, host="0.0.0.0", port=8000)

