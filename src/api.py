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
async def get_metadata(commodity: str = 'cinnamon'):
    """Return available Regions and Grades for the specified commodity"""
    # Construct path based on commodity
    commodity_file = f"{commodity}_prices.csv"
    current_data_path = os.path.join(BASE_DIR, "data", "processed", commodity_file)
    
    # Fallback to default/global if not found (e.g. for legacy reasons or if file naming differs)
    if not os.path.exists(current_data_path):
        if commodity == 'cinnamon' and os.path.exists(data_path):
             current_data_path = data_path
        else:
             print(f"DEBUG: File not found: {current_data_path}")
             # Return empty or error? Let's return empty lists so frontend doesn't crash but shows nothing
             return {
                 "commodity": commodity, 
                 "regions": [], 
                 "grades": [], 
                 "grades_by_region": {},
                 "regions_by_grade": {}
             }

    try:
        df = pd.read_csv(current_data_path)
        
        # Extract available combinations
        # grades_by_region: {"Colombo": ["Grade1", "Grade2"], ...}
        # regions_by_grade: {"Grade1": ["Colombo", "Kandy"], ...}
        series_map = {}
        reverse_map = {}
        
        if 'Region' in df.columns and 'Grade' in df.columns:
            # Get unique pairs
            pairs = df[['Region', 'Grade']].drop_duplicates()
            for _, row in pairs.iterrows():
                r, g = row['Region'], row['Grade']
                
                # Forward map (Region -> Grades)
                if r not in series_map:
                    series_map[r] = []
                series_map[r].append(g)
                
                # Reverse map (Grade -> Regions)
                if g not in reverse_map:
                    reverse_map[g] = []
                reverse_map[g].append(r)
                
        # Sort for consistency
        regions = sorted(series_map.keys())
        for r in regions:
            series_map[r].sort()
            
        # Flatten unique grades
        all_grades = sorted(list(set([g for grades in series_map.values() for g in grades])))
        
        # Sort reverse map
        for g in all_grades:
            if g in reverse_map:
                reverse_map[g].sort()

        print(f"DEBUG: Metadata for {commodity}: {len(regions)} regions, {len(all_grades)} grades")

        return {
            "commodity": commodity,
            "regions": regions,
            "grades": all_grades,
            "grades_by_region": series_map,
            "regions_by_grade": reverse_map
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest, commodity: str = 'cinnamon'):
    """Generate price forecast"""
    # Load model for specific commodity
    model = engine.load_artifacts(commodity)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model for {commodity} not found. Please train first.")
            
    # Determine data path based on commodity
    commodity_file = f"{commodity}_prices.csv"
    current_data_path = os.path.join(BASE_DIR, "data", "processed", commodity_file)
    
    if not os.path.exists(current_data_path):
        raise HTTPException(status_code=404, detail=f"Data file for {commodity} not found")

    try:
        # Load and preprocess data using engine
        # This now returns a Long format enriched dataframe (with all features)
        df = engine.load_and_prepare_data(current_data_path)
        
        # Filter logic (Region/Grade)
        if 'Grade' in df.columns:
            # Filter by Grade
            valid_grades = df['Grade'].unique()
            if request.grade not in valid_grades:
                 # Fallback: Use first available grade if requested not found? 
                 # Or specific error. Let's return error but list available.
                 raise HTTPException(status_code=400, detail=f"Grade {request.grade} not found. Available: {list(valid_grades)}")
            
            df = df[df['Grade'] == request.grade]
            
            # Filter by Region if it exists in data
            if 'Region' in df.columns:
                valid_regions = df['Region'].unique()
                if request.region in valid_regions:
                    df = df[df['Region'] == request.region]
                else:
                    # If specific region not found, maybe just warn? 
                    # For now, simplistic check.
                    pass
            
            # Ensure proper sorting
            if 'Date' in df.columns:
                df = df.sort_values('Date')

        # Get the forecast from engine
        # Note: forecast_prices now handles feature extraction dynamically
        initial_price = float(engine.forecast_prices(model, df))
        
        # Generate trend sequence starting from initial_price (Visual Mockup for now)
        dates = []
        prices = []
        last_date = df['Date'].max() if 'Date' in df.columns else datetime.datetime.now()
        current_price = initial_price
        
        for i in range(1, request.months + 1):
             next_date = last_date + datetime.timedelta(days=30 * i)
             # Slight random walk + trend
             change = np.random.normal(0, current_price * 0.02)
             current_price += float(change)
             dates.append(next_date.strftime("%Y-%m-%d"))
             prices.append(round(float(current_price), 2))

             
        return {
            "commodity": commodity,
            "region": request.region,
            "grade": request.grade,
            "forecast": {
                "dates": dates,
                "prices": prices
            },
            "history": {
                "dates": df['Date'].dt.strftime("%Y-%m-%d").tail(90).tolist() if 'Date' in df.columns else [],
                "prices": df['Regional_Price'].astype(float).tail(90).tolist() if 'Regional_Price' in df.columns else []
            }

        }

        
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
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
def get_news(commodity: str = 'cinnamon'):
    """
    Fetch market intelligence (Sentiment, Confidence, Summary).
    """
    try:
        # In a real app, you might want to cache this result as it includes web scraping
        result = get_market_intelligence(commodity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    # Reload=True can cause issues with TensorFlow on Windows
    # We pass the app object directly since reload is False (safe for script execution)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except Exception as e:
        print(f"Server error: {e}")
