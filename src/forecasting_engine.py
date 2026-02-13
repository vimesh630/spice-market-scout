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
import logging
from datetime import datetime
import json
import pickle

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

def _safe_label_transform(series, encoder, column_name):
    """Transform labels with a stable unknown fallback (0) without zeroing known labels."""
    values = series.astype(str)
    class_to_index = {str(label): idx for idx, label in enumerate(encoder.classes_)}
    encoded = values.map(class_to_index)

    # Case-insensitive fallback for mixed-casing datasets (e.g., Alba vs alba).
    unknown_mask = encoded.isna()
    if unknown_mask.any():
        class_to_index_ci = {}
        for label, idx in class_to_index.items():
            class_to_index_ci.setdefault(label.strip().casefold(), idx)
        encoded.loc[unknown_mask] = values.loc[unknown_mask].str.strip().str.casefold().map(class_to_index_ci)

    unknown_mask = encoded.isna()
    if unknown_mask.any():
        unknown_count = int(unknown_mask.sum())
        logger.warning(f"Found {unknown_count} unseen labels in {column_name}. Mapping them to 0.")
    return encoded.fillna(0).astype(int)

def preprocess_data(df, training_mode=True):
    """
    Apply feature engineering with strict encoder handling and no leakage.
    Phase 3: Classification of features (Real vs Missing).
    """
    # Convert 'Date' column to datetime objects
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort strictly by Date
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        df['Month'] = df['Date'] # Legacy support

    # Encode categorical variables
    for col in ['Grade', 'Region']:
        if col in df.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
            
            if training_mode:
                df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if not hasattr(label_encoders[col], 'classes_') or len(label_encoders[col].classes_) == 0:
                    logger.warning(f"Label encoder for {col} is not initialized. Fitting on current data as fallback.")
                    df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = _safe_label_transform(df[col], label_encoders[col], col)

    # Key Date Features
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month_num'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter

    # Lag Features (Strictly Historical)
    groups = [c for c in ['Grade', 'Region'] if c in df.columns]
    lag_columns = ['Regional_Price', 'National_Price', 'Temperature', 'Rainfall', 'Exchange_Rate']
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
        
    # Standard deviation rolling features for volatility
    for window in [12]:
        col_name = f'{col}_std_{window}'
        if groups:
            df[col_name] = df.groupby(groups)[col].transform(lambda x: x.rolling(window).std())
        else:
            df[col_name] = df[col].rolling(window).std()

    # Impute missing values without leaking future or cross-series information.
    if groups:
        sort_cols = groups + (['Date'] if 'Date' in df.columns else [])
        df = df.sort_values(sort_cols)
        fill_cols = [c for c in df.columns if c not in groups]
        df[fill_cols] = df.groupby(groups)[fill_cols].ffill()
    else:
        df = df.sort_values('Date') if 'Date' in df.columns else df
        df = df.ffill()

    # Cold-start handling: Strict 0 filling for missing lags/rolling to avoid leakage.
    # The notebook strategy is ffill then 0. 
    # We do NOT backfill from future.
    # We do NOT fill with current value (which would be leakage if current is target, 
    # though here they are features. But 'Regional_Price_lag_1' missing means we are at t=0. 
    # Filling with t=0 price implies persistence. 0 implies unknown.
    # Notebook says: "avoiding cold-start backfilling".
    # We will stick to 0 for remaining NaNs as per line 126.
    pass

    df = df.fillna(0)
    
    return df

def load_and_prepare_data(data_path, training_mode=True):
    """
    Loads and prepares data. 
    Phase 3: ZERO Randomness. Strict classification of features.
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Handle "Wide" Schema
    if 'Cinnamon_Grade_ALBA' in df.columns:
        logger.info("Detected wide schema. Transforming to long format...")
        df_melted = df.melt(id_vars=['Date', 'Market_Sentiment'], 
                           value_vars=['Cinnamon_Grade_ALBA', 'Cinnamon_Grade_C5'],
                           var_name='Grade', value_name='Regional_Price')
        df_melted['Grade'] = df_melted['Grade'].apply(lambda x: x.replace('Cinnamon_Grade_', ''))
        
        if 'Region' not in df_melted.columns:
            df_melted['Region'] = 'Colombo'
        
        # Impute National Price: Fixed spread if missing
        if 'National_Price' not in df_melted.columns:
            df_melted['National_Price'] = df_melted['Regional_Price'] * 1.15 # 15% Spread Assumption
            
        df = df_melted

    elif 'Regional_Price' in df.columns:
        if 'Region' not in df.columns: df['Region'] = 'Default'
        if 'National_Price' not in df.columns: df['National_Price'] = df['Regional_Price'] * 1.15

    # DROP Unavailable/Sparse Features
    # Only keep what we truly trust or impute consistently
    # Current strategy: Keep columns if they exist, otherwise don't mock them.
    # preprocess_data handles missingness via imputation or filling 0 if col exists.
    
    df = df.dropna(subset=['Regional_Price'])
    return preprocess_data(df, training_mode=training_mode)

def prepare_sequences(df, sequence_length=12, target_col='Regional_Price'):
    """
    Create sequences for LSTM training.
    """
    # 1. Valid Feature Selection
    potential_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD',
        'Year', 'Month_num', 'Quarter'
    ]
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    potential_features.extend(lag_cols)
    
    valid_feature_cols = [c for c in potential_features if c in df.columns]
    logger.info(f"Using {len(valid_feature_cols)} features.")

    # Fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    X_sequences, y_sequences, target_dates = [], [], []

    # Handle Series
    if 'Grade' in df.columns and 'Region' in df.columns:
        for grade in df['Grade'].unique():
            for region in df['Region'].unique():
                subset = df[(df['Grade'] == grade) & (df['Region'] == region)].sort_values('Date')
                if len(subset) < sequence_length + 1: continue

                for i in range(len(subset) - sequence_length):
                    X_seq = subset.iloc[i:i + sequence_length][valid_feature_cols].values
                    y_seq = subset.iloc[i + sequence_length][target_col]
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
                    if 'Date' in subset.columns:
                        target_dates.append(subset.iloc[i + sequence_length]['Date'])
                    else:
                        target_dates.append(i + sequence_length)
    else:
        if len(df) >= sequence_length + 1:
            for i in range(len(df) - sequence_length):
                X_seq = df.iloc[i:i + sequence_length][valid_feature_cols].values
                y_seq = df.iloc[i + sequence_length][target_col]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                if 'Date' in df.columns:
                    target_dates.append(df.iloc[i + sequence_length]['Date'])
                else:
                    target_dates.append(i + sequence_length)

    return np.array(X_sequences), np.array(y_sequences), valid_feature_cols, np.array(target_dates)

def build_model(input_shape, model_type='GRU'):
    """
    Builds the GRU/LSTM model based on notebook architecture.
    """
    model = Sequential()
    
    if model_type == 'GRU':
        # GRU Architecture from Notebook (Trial 0-ish High Performance or Default)
        model.add(GRU(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False))
        model.add(Dropout(0.2))
    else:
        # LSTM Fallback
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Optimizer from notebook defaults
    opt = Adam(learning_rate=0.001)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

def save_model(model, history, results, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'lstm_model.keras'))
    with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f: pickle.dump(history.history, f)
    with open(os.path.join(model_dir, 'scaler_features.pkl'), 'wb') as f: pickle.dump(scaler_features, f)
    with open(os.path.join(model_dir, 'scaler_target.pkl'), 'wb') as f: pickle.dump(scaler_target, f)
    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f: pickle.dump(label_encoders, f)
    if 'feature_cols' in results:
         with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f: json.dump(results['feature_cols'], f)

def calculate_directional_accuracy(y_true, y_pred):
    """Phase 3: Realism Metric - Directional Accuracy"""
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    # Align lengths
    min_len = min(len(diff_true), len(diff_pred))
    if min_len == 0:
        return 0.0
    correct_direction = np.sign(diff_true[:min_len]) == np.sign(diff_pred[:min_len])
    return np.mean(correct_direction)

def train_model(df, commodity='cinnamon', epochs=50, batch_size=32, **kwargs):
    """
    Train model using PHASE 3 Strict CHRONOLOGICAL split (70/15/15).
    """
    global scaler_features, scaler_target
    
    scaler_features = StandardScaler()
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    # 1. Prepare Sequences
    X, y, feature_cols, target_dates = prepare_sequences(df, SEQUENCE_LENGTH)
    
    if len(X) == 0: raise ValueError("No sequences created.")

    # Strict chronological order across all sequences before splitting.
    sort_idx = np.argsort(target_dates)
    X = X[sort_idx]
    y = y[sort_idx]

    # 2. Split (before scaling to avoid leakage)
    idx_train = int(len(X) * 0.70)
    idx_val = int(len(X) * 0.85)

    X_train_raw, y_train_raw = X[:idx_train], y[:idx_train]
    X_val_raw, y_val_raw = X[idx_train:idx_val], y[idx_train:idx_val]
    X_test_raw, y_test_raw = X[idx_val:], y[idx_val:]

    if len(X_train_raw) == 0 or len(X_val_raw) == 0 or len(X_test_raw) == 0:
        raise ValueError("Insufficient sequences for train/val/test split.")

    # 3. Scale (fit only on train split)
    n_samples, n_timesteps, n_features = X.shape
    scaler_features.fit(X_train_raw.reshape(-1, n_features))
    X_train = scaler_features.transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape[0], n_timesteps, n_features)
    X_val = scaler_features.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape[0], n_timesteps, n_features)
    X_test = scaler_features.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape[0], n_timesteps, n_features)

    scaler_target.fit(y_train_raw.reshape(-1, 1))
    y_train = scaler_target.transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = scaler_target.transform(y_val_raw.reshape(-1, 1)).flatten()
    y_test = scaler_target.transform(y_test_raw.reshape(-1, 1)).flatten()

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Train
    input_shape = (X_train.shape[1], X_train.shape[2])
    # GRU is the requested architecture
    model = build_model(input_shape, model_type='GRU')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        shuffle=False,
        verbose=1
    )

    # 5. Evaluate Realism (Phase 3)
    y_pred_test = model.predict(X_test).flatten()
    y_true_orig = y_test_raw
    y_pred_orig = scaler_target.inverse_transform(y_pred_test.reshape(-1,1)).flatten()
    
    da = calculate_directional_accuracy(y_true_orig, y_pred_orig)
    logger.info(f"Directional Accuracy on Test: {da:.2%}")

    # 6. Save Results
    results = {
        'feature_cols': feature_cols,
        'directional_accuracy': da
    }
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    save_model(model, history, results, model_dir)
    
    return model, history, results

def load_artifacts(commodity='cinnamon'):
    global scaler_features, scaler_target, label_encoders
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    try:
        model = load_model(os.path.join(model_dir, 'lstm_model.keras'))
        with open(os.path.join(model_dir, 'scaler_features.pkl'), 'rb') as f: scaler_features = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_target.pkl'), 'rb') as f: scaler_target = pickle.load(f)
        enc_path = os.path.join(model_dir, 'label_encoders.pkl')
        if os.path.exists(enc_path):
             with open(enc_path, 'rb') as f: label_encoders = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None

def _predict_single_step(model, df_sequence, train_features):
    df_sequence = df_sequence.copy()
    # Filter/Fill
    for col in train_features:
        if col not in df_sequence.columns: df_sequence[col] = 0
    
    X_seq = df_sequence[train_features].values
    if X_seq.shape[0] < SEQUENCE_LENGTH:
         pad_len = SEQUENCE_LENGTH - X_seq.shape[0]
         padding = np.repeat(X_seq[0].reshape(1,-1), pad_len, axis=0)
         X_seq = np.vstack([padding, X_seq])
    
    X_seq = X_seq[-SEQUENCE_LENGTH:]
    
    # Scale
    X_flat = X_seq.reshape(-1, len(train_features))
    if hasattr(scaler_features, 'n_features_in_') and X_flat.shape[1] != scaler_features.n_features_in_:
        return None # Mismatch

    X_scaled = scaler_features.transform(X_flat).reshape(1, SEQUENCE_LENGTH, len(train_features))
    pred_scaled = model.predict(X_scaled, verbose=0)
    return scaler_target.inverse_transform(pred_scaled)[0][0]

def forecast_multistep(model, df, steps=24, commodity='cinnamon'):
    """
    PHASE 3: SCENARIO FORECASTING + ADAPTIVE CLAMPING + ANCHORING
    Returns a dictionary of scenarios: {'Baseline': ..., 'Optimistic': ..., 'Pessimistic': ...}
    """
    logger.info(f"Generating Phase 3 Scenarios for {commodity}...")
    
    # Setup
    start_df = df.copy()
    if 'Date' in start_df.columns: 
        start_df = start_df.sort_values('Date')
    
    # Load feature alignment
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    try:
        with open(os.path.join(model_dir, 'feature_cols.json'), 'r') as f: train_features = json.load(f)
    except:
        train_features = [] # Fallback

    # Determine Volatility for Adaptive Clamp
    # We use the std of the last 12 months as a proxy for "normal volatility"
    last_real_price = start_df['Regional_Price'].iloc[-1]
    last_12_prices = start_df['Regional_Price'].iloc[-12:]
    rolling_std = last_12_prices.std()
    
    if pd.isna(rolling_std) or rolling_std == 0: 
        rolling_std = last_real_price * 0.05
    
    adaptive_clamp_range = 2.5 * rolling_std 
    logger.info(f"Adaptive Clamp set to +/- {adaptive_clamp_range:.2f} (Based on 12m StdDev: {rolling_std:.2f})")

    # --- ANCHORING ---
    # Calculate bias correction to prevent jump at t=0
    # Simulate t+1 with naive assumption (flat) just to check model offset
    temp_df = start_df.copy()
    next_date = temp_df.iloc[-1]['Date'] + pd.DateOffset(months=1)
    next_row = temp_df.iloc[-1].copy()
    next_row['Date'] = next_date
    # Naive assumption for t+1 features: same as t
    temp_df = pd.concat([temp_df, pd.DataFrame([next_row])], ignore_index=True)
    temp_df = preprocess_data(temp_df, training_mode=False) # Helper to fill lags
    
    raw_pred_t1 = _predict_single_step(model, temp_df.iloc[-SEQUENCE_LENGTH:], train_features)
    if raw_pred_t1 is None: raw_pred_t1 = last_real_price

    # Bias = Real - Model. If Model says 100 but Real is 110, Bias is +10.
    # We add (+10) to future predictions to shift them up.
    anchor_bias = last_real_price - raw_pred_t1
    logger.info(f"Anchoring Bias: {anchor_bias:.2f}")

    # Scenarios Definition
    # Dynamic Scenario Parameters based on Commodity
    if commodity.lower() == 'pepper':
        # Aggressive Bullish Parameters for Pepper to counteract bias
        scenarios = {
            'Baseline': {
                'drift': 1.002,          # Neutral/Slight Growth (was default 1.002)
                'supply_mod': 1.00,
                'demand_mod': 1.00,
                'weather_amp': 1.0
            },
            'Optimistic': {
                'drift': 1.002,          # Neutral (Prevent Export Competition Bias)
                'supply_mod': 0.90,      # Severe Scarcity
                'demand_mod': 1.10,      # High Demand
                'weather_amp': 0.8
            }, 
            'Pessimistic': {
                'drift': 0.995,          # Weak Market
                'supply_mod': 1.05,
                'demand_mod': 0.95,
                'weather_amp': 1.2
            } 
        }
        logger.info("Using Aggressive Bullish Scenarios for Pepper.")
    else:
        # Standard Parameters for Cinnamon/Clove
        scenarios = {
            'Baseline': {
                'drift': 1.002,
                'supply_mod': 1.00,
                'demand_mod': 1.00,
                'weather_amp': 1.0
            },
            'Optimistic': {
                'drift': 1.005,
                'supply_mod': 0.95,
                'demand_mod': 1.05,
                'weather_amp': 0.8
            }, 
            'Pessimistic': {
                'drift': 0.995,
                'supply_mod': 1.05,
                'demand_mod': 0.95,
                'weather_amp': 1.2
            } 
        }

    results = {}

    for name, params in scenarios.items():
        current_df = start_df.copy()
        scenario_dates = []
        scenario_prices = []
        
        # Spread Logic
        last_row = current_df.iloc[-1]
        current_spread = 0
        if 'National_Price' in last_row:
             current_spread = last_row['National_Price'] - last_row['Regional_Price']

        for i in range(1, steps + 1):
            # 1. Advance Time
            last_date = current_df.iloc[-1]['Date']
            next_date = last_date + pd.DateOffset(months=1)
            month_num = next_date.month
            
            # 2. Base Exogenous (Sine Waves & Drifts)
            next_row = current_df.iloc[-1].copy()
            next_row['Date'] = next_date
            next_row['Year'] = next_date.year
            next_row['Month'] = next_date # For legacy
            next_row['Month_num'] = month_num
            next_row['Quarter'] = next_date.quarter
            
            # Weather Simulation (Sine Wave with Inter-annual Randomness approx)
            # Peak Temp in April/May (Month 4/5), Lowest in Dec/Jan
            # Sri Lanka: Warm year round, but slight fluctuation.
            # Base 27C +/- 1.5C.
            temp_wave = np.sin((month_num - 4) * np.pi / 6) 
            next_row['Temperature'] = 27 + (1.5 * temp_wave * params['weather_amp'])
            
            # Rainfall: Rainy Seasons (May-June, Oct-Nov). Complex, but lets approximate.
            # Peaks at 5 and 10.
            rain_wave = np.sin((month_num - 5) * np.pi / 3) # Faster cycle?
            next_row['Rainfall'] = 200 + (100 * rain_wave * params['weather_amp'])

            # --- 3. Production & Consumption Logic (Re-Added) ---
            # Base Production
            base_prod = current_df.iloc[-1].get('Local_Production_Volume', 1000)
            
            # Harvest Spikes (Seasonality)
            if commodity.lower() == 'pepper':
                if month_num in [5, 6, 7]: base_prod *= 1.3 # Harvest Peak
            elif commodity.lower() == 'cinnamon':
                if month_num in [5, 6, 11, 12]: base_prod *= 1.2
            elif commodity.lower() == 'clove':
                if month_num in [12, 1, 2]: base_prod *= 1.3

            # Apply Scenario Modifiers (The Fix)
            if 'Local_Production_Volume' in next_row:
                next_row['Local_Production_Volume'] = base_prod * params['supply_mod']
            
            if 'Global_Consumption_Volume' in next_row:
                next_row['Global_Consumption_Volume'] = next_row['Global_Consumption_Volume'] * params['demand_mod']

            # Competitors / Global (Drift)
            drift_cols = ['Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Exchange_Rate', 'Inflation_Rate']
            for col in drift_cols:
                if col in next_row:
                    next_row[col] = next_row[col] * params['drift']
            
            # 3. Append to History & Recalculate Features
            # We must append BEFORE predicting to get Lags/Rolling for 't'
            current_df = pd.concat([current_df, pd.DataFrame([next_row])], ignore_index=True)
            
            # Re-run preprocess to fill lags/rolling for the NEW last row
            # Note: This is computationally creating the whole history again, but safe.
            # Only need tail for prediction.
            current_df = preprocess_data(current_df, training_mode=False)
            
            # 4. Predict
            input_sequence = current_df.iloc[-SEQUENCE_LENGTH:]
            raw_pred = _predict_single_step(model, input_sequence, train_features)
            
            if raw_pred is None:
                # Fallback
                raw_pred = current_df.iloc[-2]['Regional_Price']

            # 5. Apply Logic
            # Anchor Decay: We trust the model more as we go further out?
            # Or constant bias? Let's use constant bias for stability.
            adjusted_pred = raw_pred + anchor_bias
            
            # Adaptive Clamp (relative to PREVIOUS step)
            prev_price = current_df.iloc[-2]['Regional_Price']
            
            # Allow max change based on historical volatility OR percentage cap (whichever is tighter)
            # This prevents massive jumps if volatility was high, but also allows movement.
            # We use 20% cap as a failsafe.
            max_step_change = adaptive_clamp_range
            if prev_price > 0:
                max_step_change = min(adaptive_clamp_range, prev_price * 0.20)
            
            upper_bound = prev_price + max_step_change
            lower_bound = prev_price - max_step_change
            
            clamped_pred = np.clip(adjusted_pred, lower_bound, upper_bound)
            
            # Scenario Nudges (Explicit Manual Override)
            if name == 'Optimistic':
                if commodity.lower() == 'pepper':
                    clamped_pred *= 1.008 # Boosted Nudge for Pepper (0.8% per month)
                else:
                    clamped_pred *= 1.002 # Subtle cumulative boost
            elif name == 'Pessimistic':
                clamped_pred *= 0.998 # Subtle cumulative drag
            
            # --- HARD FLOOR SUPPORT (The Fix) ---
            # If price drops below 70% of last real price, force it to stay there.
            support_level = last_real_price * 0.70
            if clamped_pred < support_level:
                clamped_pred = support_level

            # Hard Floor at 0
            final_pred = float(max(0.0, clamped_pred))

            # 6. Update the 'Truth' in the DataFrame for the next step
            # The row we added had 'Regional_Price' from the previous step (copy). 
            # We must update it to the PREDICTED value so next lag_1 is correct.
            current_df.iloc[-1, current_df.columns.get_loc('Regional_Price')] = final_pred
            
            if 'National_Price' in current_df.columns:
                 current_df.iloc[-1, current_df.columns.get_loc('National_Price')] = final_pred + current_spread

            scenario_dates.append(next_date.strftime("%Y-%m-%d"))
            scenario_prices.append(float(final_pred))
            
        results[name] = {'dates': scenario_dates, 'prices': scenario_prices}
    
    # "Save" these forecast scenarios to the file system or return?
    # The API calls this, so returning dict is fine.
    
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commodities = ['cinnamon', 'clove', 'pepper']
    for com in commodities:
        data_path = os.path.join(base_dir, 'data', 'processed', f'{com}_prices.csv')
        if os.path.exists(data_path):
            print(f"\n--- Training {com} Model ---")
            df = load_and_prepare_data(data_path)
            # Train for production (30 epochs with EarlyStopping is usually sufficient)
            model, history, results = train_model(df, commodity=com, epochs=30) 
            
            print(f"Generating Verification Forecast...")
            forecasts = forecast_multistep(model, df, steps=12, commodity=com)
            
            # Print a quick summary
            print(f"Forecast Summary (Next 3 Months - Baseline):")
            baseline = forecasts.get('Baseline', {})
            prices = baseline.get('prices', [])[:3]
            print(f"  {prices}")
            
            print(f"✅ {com} model trained and saved.")
        else:
            print(f"Skipping {com}, file not found at {data_path}")
