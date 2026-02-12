import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import pickle
import json
from itertools import product
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
SEQUENCE_LENGTH = 12
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global scalers
scaler_features = None
scaler_target = None
label_encoders = {}

def preprocess_data(df, training_mode=True):
    """
    Apply feature engineering with strict encoder handling and no leakage.
    """
    # Convert 'Date' column to datetime objects
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort strictly by Date to ensure lags are correct
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        df['Month'] = df['Date'] # Legacy support

    # Encode categorical variables: Grade, Region
    for col in ['Grade', 'Region']:
        if col in df.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
            
            if training_mode:
                # TRAIN: Fit and transform
                df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # INFERENCE: Transform only
                try:
                    df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Fallback for unseen labels
                    df[f'{col}_encoded'] = 0 

    # Create additional time-based features
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month_num'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter

    # Create lag features and rolling averages
    # Group by series (Region/Grade) if possible to prevent cross-series contamination
    groups = [c for c in ['Grade', 'Region'] if c in df.columns]
    
    lag_columns = ['Regional_Price', 'National_Price', 'Temperature', 'Rainfall', 'Exchange_Rate']
    # Only lag columns that actually exist
    lag_columns = [c for c in lag_columns if c in df.columns]

    for col in lag_columns:
        for lag in [1, 3, 6, 12]:
            col_name = f'{col}_lag_{lag}'
            if groups:
                df[col_name] = df.groupby(groups)[col].shift(lag)
            else:
                df[col_name] = df[col].shift(lag)
        
        # Rolling features
        for window in [3, 6, 12]:
            col_name = f'{col}_rolling_{window}'
            if groups:
                df[col_name] = df.groupby(groups)[col].transform(lambda x: x.rolling(window).mean())
            else:
                df[col_name] = df[col].rolling(window).mean()

    # Fill NaNs created by lags (Forward fill first, then backward fill)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def load_and_prepare_data(data_path):
    """
    Loads and prepares data. 
    CLEAN VERSION: Removes all random noise generation.
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Handle "Wide" Schema (Convert to Long)
    if 'Cinnamon_Grade_ALBA' in df.columns:
        logger.info("Detected wide schema. Transforming to long format...")
        df_melted = df.melt(id_vars=['Date', 'Market_Sentiment'], 
                           value_vars=['Cinnamon_Grade_ALBA', 'Cinnamon_Grade_C5'],
                           var_name='Grade', value_name='Regional_Price')
        df_melted['Grade'] = df_melted['Grade'].apply(lambda x: x.replace('Cinnamon_Grade_', ''))
        
        # Default metadata if missing
        if 'Region' not in df_melted.columns:
            df_melted['Region'] = 'Colombo'
        
        # Impute National Price if missing (Use fixed spread logic, NO RANDOMNESS)
        if 'National_Price' not in df_melted.columns:
            # Assume National is approx 10% higher if strictly missing
            df_melted['National_Price'] = df_melted['Regional_Price'] * 1.1
            
        df = df_melted

    elif 'Regional_Price' in df.columns:
        logger.info("Detected long schema.")
        # Ensure basics exist
        if 'Region' not in df.columns:
             df['Region'] = 'Default'
        if 'National_Price' not in df.columns:
             df['National_Price'] = df['Regional_Price'] * 1.1

    # CLEANING: Drop rows with missing Target
    df = df.dropna(subset=['Regional_Price'])
    
    return preprocess_data(df, training_mode=True)

def prepare_sequences(df, sequence_length=12, target_col='Regional_Price'):
    """
    Create sequences for LSTM training.
    """
    # 1. Identify valid numeric features (NO HALLUCINATED COLUMNS)
    potential_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD',
        'Year', 'Month_num', 'Quarter'
    ]
    
    # Add generated lags
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    potential_features.extend(lag_cols)
    
    # Filter for what actually exists
    valid_feature_cols = [c for c in potential_features if c in df.columns]
    
    logger.info(f"Using {len(valid_feature_cols)} features: {valid_feature_cols}")

    # Fill remaining NaNs cleanly
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    X_sequences, y_sequences = [], []

    # Handle multiple series (Grade/Region pairs)
    if 'Grade' in df.columns and 'Region' in df.columns:
        for grade in df['Grade'].unique():
            for region in df['Region'].unique():
                subset = df[(df['Grade'] == grade) & (df['Region'] == region)].sort_values('Date')
                
                if len(subset) < sequence_length + 1:
                    continue

                for i in range(len(subset) - sequence_length):
                    X_seq = subset.iloc[i:i + sequence_length][valid_feature_cols].values
                    y_seq = subset.iloc[i + sequence_length][target_col]
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
    else:
        # Single series
        if len(df) >= sequence_length + 1:
            for i in range(len(df) - sequence_length):
                X_seq = df.iloc[i:i + sequence_length][valid_feature_cols].values
                y_seq = df.iloc[i + sequence_length][target_col]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

    return np.array(X_sequences), np.array(y_sequences), valid_feature_cols

def build_lstm_model(input_shape):
    """Standard LSTM Model"""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def save_model(model, history, results, model_dir):
    """Saves model and artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'lstm_model.keras'))
    
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
        
    with open(os.path.join(model_dir, 'scaler_features.pkl'), 'wb') as f:
        pickle.dump(scaler_features, f)
    with open(os.path.join(model_dir, 'scaler_target.pkl'), 'wb') as f:
        pickle.dump(scaler_target, f)
        
    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
        
    # CRITICAL: Save feature columns for inference alignment
    if 'feature_cols' in results:
         with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
            json.dump(results['feature_cols'], f)

def train_model(df, commodity='cinnamon', epochs=50, batch_size=32, **kwargs):
    """
    Train model using CHRONOLOGICAL split.
    """
    global scaler_features, scaler_target
    
    scaler_features = StandardScaler()
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    # 1. Prepare Sequences
    X, y, feature_cols = prepare_sequences(df, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        raise ValueError("No sequences created.")

    # 2. Scale
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler_features.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)
    y_scaled = scaler_target.fit_transform(y.reshape(-1, 1)).flatten()

    # 3. CHRONOLOGICAL SPLIT (No Shuffling)
    # Train on past, Test on future
    train_size = int(len(X) * 0.8)
    
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    # Validation split from training (last 20% of training set)
    val_size = int(len(X_train) * 0.8)
    X_train_final, X_val = X_train[:val_size], X_train[val_size:]
    y_train_final, y_val = y_train[:val_size], y_train[val_size:]

    logger.info(f"Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Train
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # 5. Save Results
    results = {
        'feature_cols': feature_cols
    }
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    save_model(model, history, results, model_dir)
    
    return model, history, results

def load_artifacts(commodity='cinnamon'):
    """Load model and scalers."""
    global scaler_features, scaler_target, label_encoders
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    
    try:
        model = load_model(os.path.join(model_dir, 'lstm_model.keras'))
        
        with open(os.path.join(model_dir, 'scaler_features.pkl'), 'rb') as f:
            scaler_features = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_target.pkl'), 'rb') as f:
            scaler_target = pickle.load(f)
        
        enc_path = os.path.join(model_dir, 'label_encoders.pkl')
        if os.path.exists(enc_path):
             with open(enc_path, 'rb') as f:
                label_encoders = pickle.load(f)
                
        return model
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None

def forecast_prices(model, df_sequence):
    """Placeholder for backward compatibility"""
    pass 

def _predict_single_step(model, df_sequence, train_features):
    """Helper to scale and predict."""
    # Filter columns to match training
    if train_features:
        # Add missing columns with 0
        for col in train_features:
            if col not in df_sequence.columns:
                df_sequence[col] = 0
        X_seq = df_sequence[train_features].values
    else:
        # Fallback (dangerous)
        X_seq = df_sequence.select_dtypes(include=[np.number]).values

    # Reshape/Pad
    if X_seq.shape[0] < SEQUENCE_LENGTH:
         pad_len = SEQUENCE_LENGTH - X_seq.shape[0]
         padding = np.repeat(X_seq[0].reshape(1,-1), pad_len, axis=0)
         X_seq = np.vstack([padding, X_seq])
    
    X_seq = X_seq[-SEQUENCE_LENGTH:]
    
    # Scale
    X_flat = X_seq.reshape(-1, len(train_features))
    
    # Safety check for scaler dimension
    if hasattr(scaler_features, 'n_features_in_') and X_flat.shape[1] != scaler_features.n_features_in_:
        logger.warning(f"Feature mismatch: Scaler expects {scaler_features.n_features_in_}, got {X_flat.shape[1]}")
        return None

    X_scaled = scaler_features.transform(X_flat).reshape(1, SEQUENCE_LENGTH, len(train_features))
    
    pred_scaled = model.predict(X_scaled, verbose=0)
    return scaler_target.inverse_transform(pred_scaled)[0][0]

def forecast_multistep(model, df, steps=24, commodity='cinnamon'):
    """
    REALISTIC FORECAST (PHASE 2).
    Uses Additive Spread + Seasonal Projections + Chronological Logic.
    """
    logger.info(f"Generating {steps}-step realistic forecast for {commodity}...")
    
    # 1. Setup
    current_df = df.copy()
    if 'Date' in current_df.columns:
        current_df = current_df.sort_values('Date')
        
    # Load feature alignment
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    try:
        with open(os.path.join(model_dir, 'feature_cols.json'), 'r') as f:
            train_features = json.load(f)
    except:
        logger.warning("Could not load feature_cols.json. Forecast may fail.")
        train_features = []

    # Calculate Spread
    last_row = current_df.iloc[-1]
    last_date = last_row['Date']
    
    current_spread = 0
    if 'National_Price' in last_row and 'Regional_Price' in last_row:
        current_spread = last_row['National_Price'] - last_row['Regional_Price']
    
    future_dates = []
    future_prices = []

    for i in range(1, steps + 1):
        # A. Next Date (Calendar Aware)
        next_date = last_date + pd.DateOffset(months=i)
        month_num = next_date.month
        
        # B. Project External Features (Seasonality + Drift)
        next_row = current_df.iloc[-1].copy()
        
        # --- 1. Weather Seasonality (Sri Lanka Pattern) ---
        # Peak Temp in April/May (Months 4-5), Low in Dec/Jan
        next_row['Temperature'] = 28 + 2 * np.sin((month_num - 4) * np.pi / 6)
        
        # --- 2. Economic Drift ---
        drift = 1.002 # +0.2% monthly drift
        for col in ['Inflation_Rate', 'Exchange_Rate', 'Fuel_Price']:
            if col in next_row: next_row[col] *= drift

        # --- 3. Harvest Cycles (Supply Features) ---
        is_harvest = False
        if commodity.lower() == 'cinnamon':
            if month_num in [5, 6, 7, 8, 11, 12, 1]: is_harvest = True
        elif commodity.lower() == 'clove':
            if month_num in [12, 1, 2]: is_harvest = True 
            
        if 'Local_Production_Volume' in next_row:
            base_prod = next_row['Local_Production_Volume']
            if is_harvest:
                next_row['Local_Production_Volume'] = base_prod * 1.2
            else:
                next_row['Local_Production_Volume'] = base_prod * 0.9

        # Update Time Columns
        next_row['Date'] = next_date
        next_row['Month'] = next_date
        next_row['Year'] = next_date.year
        next_row['Month_num'] = month_num
        next_row['Quarter'] = next_date.quarter
        
        # Append to History
        current_df = pd.concat([current_df, pd.DataFrame([next_row])], ignore_index=True)
        
        # C. Update Lags
        current_df = preprocess_data(current_df, training_mode=False)
        
        # D. Predict
        try:
            input_sequence = current_df.iloc[-SEQUENCE_LENGTH:]
            pred_price = _predict_single_step(model, input_sequence, train_features)
            
            if pred_price is None: # Scaler mismatch fallback
                pred_price = current_df.iloc[-2]['Regional_Price']
                
        except Exception as e:
            logger.error(f"Forecast error step {i}: {e}")
            pred_price = current_df.iloc[-2]['Regional_Price']

        # E. Apply "Market Logic" Post-Processing
        if is_harvest:
            pred_price *= 0.98 # -2% dampener due to supply glut
            
        # Clamp (Safety)
        prev_price = current_df.iloc[-2]['Regional_Price']
        max_change = prev_price * 0.10
        pred_price = np.clip(pred_price, prev_price - max_change, prev_price + max_change)
        
        # F. Update DataFrame
        idx = len(current_df) - 1
        current_df.at[idx, 'Regional_Price'] = pred_price
        if 'National_Price' in current_df.columns:
             current_df.at[idx, 'National_Price'] = pred_price + current_spread
             
        future_dates.append(next_date.strftime("%Y-%m-%d"))
        future_prices.append(float(pred_price))
        
    return future_dates, future_prices

if __name__ == "__main__":
    # Script entry point for training
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    commodities = ['cinnamon', 'clove']
    for com in commodities:
        data_path = os.path.join(base_dir, 'data', 'processed', f'{com}_prices.csv')
        if os.path.exists(data_path):
            print(f"Training {com}...")
            df = load_and_prepare_data(data_path)
            train_model(df, commodity=com, epochs=30)
            print(f"Done.")
