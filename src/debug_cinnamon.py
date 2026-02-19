
import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define paths
# Script is in src/, so parent is root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'lstm_cinnamon')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cinnamon_prices.csv')

def load_and_debug():
    print(f"--- Debugging Cinnamon Model at {MODEL_DIR} ---")
    
    # 1. Load Scalers
    try:
        with open(os.path.join(MODEL_DIR, 'scaler_target.pkl'), 'rb') as f:
            scaler_target = pickle.load(f)
        print(f"✅ Loaded scaler_target: {type(scaler_target)}")
        if hasattr(scaler_target, 'mean_'):
            print(f"   Mean: {scaler_target.mean_}, Scale: {scaler_target.scale_}")
        elif hasattr(scaler_target, 'min_'):
            print(f"   Min: {scaler_target.min_}, Scale: {scaler_target.scale_}")
    except Exception as e:
        print(f"❌ Failed to load scaler_target: {e}")
        return

    # 2. Load Model
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'), compile=False)
        print(f"✅ Loaded model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Predict on dummy data
    # Create a dummy input (1, 12, 59) - assuming 59 features
    input_shape = model.input_shape
    print(f"   Model Input Shape: {input_shape}")
    
    dummy_input = np.zeros((1, 12, input_shape[2]))
    print(f"   Predicting on zero input...")
    pred_raw = model.predict(dummy_input)
    print(f"   Raw Prediction: {pred_raw}")
    
    pred_inv = scaler_target.inverse_transform(pred_raw)
    print(f"   Inverse Transformed Prediction: {pred_inv}")
    
    if pred_inv[0][0] == 0:
        print("❌ PREDICTION IS ZERO!")
    else:
        print("✅ Prediction is non-zero.")

if __name__ == "__main__":
    load_and_debug()
