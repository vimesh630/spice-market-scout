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

    # Impute missing values without leaking future or cross-series information.
    if groups:
        sort_cols = groups + (['Date'] if 'Date' in df.columns else [])
        df = df.sort_values(sort_cols)
        fill_cols = [c for c in df.columns if c not in groups]
        df[fill_cols] = df.groupby(groups)[fill_cols].ffill()
    else:
        df = df.sort_values('Date') if 'Date' in df.columns else df
        df = df.ffill()

    # Cold-start lag/rolling values should not be backfilled from future rows.
    for col in lag_columns:
        for lag in [1, 3, 6, 12]:
            lag_col = f'{col}_lag_{lag}'
            if lag_col in df.columns:
                df[lag_col] = df[lag_col].fillna(df[col])
        for window in [3, 6, 12]:
            roll_col = f'{col}_rolling_{window}'
            if roll_col in df.columns:
                df[roll_col] = df[roll_col].fillna(df[col])

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

def build_lstm_model(input_shape):
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
    model = build_lstm_model(input_shape)
    
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
    last_real_price = start_df['Regional_Price'].iloc[-1]
    rolling_std = start_df['Regional_Price'].rolling(12).std().iloc[-1]
    if pd.isna(rolling_std) or rolling_std == 0: 
        rolling_std = last_real_price * 0.05
    
    adaptive_clamp = 2.5 * rolling_std # 2.5 Sigma
    logger.info(f"Adaptive Clamp set to +/- {adaptive_clamp:.2f}")

    # --- ANCHORING STEP ---
    # We generate a "Test Prediction" for t+1 using the raw model state
    # And compare it to what we expect (continuity).
    # However, since we are doing iterative, we can just calculate the OFFSET
    # required to make the first prediction match the last real price (or be very close).
    
    # Actually, a better approach for "Realism" is to let the first prediction happen,
    # calculate the Delta (Pred_t1 - Last_Real), and if it's huge, we subtract it.
    # But strictly speaking, we want: Forecast_t = Raw_Model_t + Bias
    # Where Bias = Last_Real - Raw_Model_Reconstruction_of_Last_Real?
    # Or simpler: Bias = Last_Real - Raw_Model_Prediction_t1 (This forces t1 ~= Last_Real, i.e. 0 growth).
    # Let's try: Bias = Last_Real - (Raw_Model_t1_Prediction).
    # Then Adjusted_t1 = Raw_Model_t1 + Bias = Last_Real. 
    # This forces continuity. Then Adjusted_t2 = Raw_Model_t2 + Bias.
    # This preserves the *shape* and *trend* of the model but shifts the level.
    
    # Let's get the raw prediction for t+1 first (using Baseline inputs)
    temp_df = start_df.copy()
    # Add dummy row for t+1
    next_date = temp_df.iloc[-1]['Date'] + pd.DateOffset(months=1)
    next_row = temp_df.iloc[-1].copy()
    next_row['Date'] = next_date
    temp_df = pd.concat([temp_df, pd.DataFrame([next_row])], ignore_index=True)
    temp_df = preprocess_data(temp_df, training_mode=False)
    
    raw_pred_t1 = _predict_single_step(model, temp_df.iloc[-SEQUENCE_LENGTH:], train_features)
    if raw_pred_t1 is None: 
         raw_pred_t1 = last_real_price

    anchor_bias = last_real_price - raw_pred_t1
    # anchor_bias = float(np.clip(anchor_bias, -adaptive_clamp, adaptive_clamp))
    logger.info(f"Anchoring Bias Calculated: {anchor_bias:.2f} (Last Real: {last_real_price:.2f} vs Raw Pred: {raw_pred_t1:.2f})")

    # Scenarios Definition
    # Optimistic: Price GOES UP (High Demand, Low Supply)
    # Pessimistic: Price GOES DOWN (Low Demand, High Supply)
    scenarios = {
        'Baseline': {
            'drift': 1.002, 
            'supply_mod': 1.00,
            'demand_mod': 1.00,
            'weather_mod': 1.0
        },
        'Optimistic': {
            'drift': 1.005, # Inflation helps price
            'supply_mod': 0.95, # Scarcity
            'demand_mod': 1.05, # High Demand
            'weather_mod': 0.95 # Mild weather (good quality?) Or bad weather (scarcity?) -> Let's assume Scarcity = High Price
        }, 
        'Pessimistic': {
            'drift': 0.995, # Deflation?
            'supply_mod': 1.05, # Oversupply
            'demand_mod': 0.95, # Low Demand
            'weather_mod': 1.1 # Volatile weather but somehow low price? Maybe poor quality.
        } 
    }

    results = {}

    for name, params in scenarios.items():
        current_df = start_df.copy()
        
        # Spread Logic
        last_row = current_df.iloc[-1]
        current_spread = 0
        if 'National_Price' in last_row:
             current_spread = last_row['National_Price'] - last_row['Regional_Price']

        scenario_dates = []
        scenario_prices = []

        for i in range(1, steps + 1):
            next_date = current_df.iloc[-1]['Date'] + pd.DateOffset(months=1)
            month_num = next_date.month
            
            # Project Exogenous
            next_row = current_df.iloc[-1].copy()
            
            # Weather Sine Wave
            base_temp = 28 + 2 * np.sin((month_num - 4) * np.pi / 6)
            next_row['Temperature'] = base_temp * params['weather_mod']
            
            # Economic/Supply/Demand Drift
            if 'Inflation_Rate' in next_row: next_row['Inflation_Rate'] *= params['drift']
            if 'Global_Consumption_Volume' in next_row: next_row['Global_Consumption_Volume'] *= params['demand_mod']
            if 'Local_Production_Volume' in next_row: next_row['Local_Production_Volume'] *= params['supply_mod']

            # Update Time
            next_row['Date'] = next_date
            next_row['Month'] = next_date
            next_row['Year'] = next_date.year
            next_row['Month_num'] = month_num
            next_row['Quarter'] = next_date.quarter
            
            # Append & Recalc Lags
            current_df = pd.concat([current_df, pd.DataFrame([next_row])], ignore_index=True)
            current_df = preprocess_data(current_df, training_mode=False)
            
            # Predict
            try:
                input_sequence = current_df.iloc[-SEQUENCE_LENGTH:]
                raw_pred = _predict_single_step(model, input_sequence, train_features)
                if raw_pred is None: raw_pred = current_df.iloc[-2]['Regional_Price']
            except:
                 raw_pred = current_df.iloc[-2]['Regional_Price']

            # Apply Anchoring
            # We decay the bias slowly over time to eventually trust the model? 
            # Or keep it constant to maintain level shift?
            # For 12 months, constant is safer to prevent reversion to "bad" mean.
            # Decay anchoring gradually so long-horizon forecasts are model-led.
            bias_decay = np.exp(-0.05 * (i - 1))
            adjusted_pred = raw_pred + (anchor_bias * bias_decay)

            # Adaptive Clamp (relative to PREVIOUS step)
            prev_price = current_df.iloc[-2]['Regional_Price']
            
            # We allow the trend, but clamp the *change*
            # But wait, adjusted_pred might be huge jump if model is wild.
            # Clamp change from prev_price
            
            max_step_change = min(adaptive_clamp, prev_price * 0.12)
            final_pred = np.clip(adjusted_pred, prev_price - max_step_change, prev_price + max_step_change)
            
            # Scenario specific manual nudges (Post-Model)
            # This ensures differentiation even if model ignores features
            if name == 'Optimistic':
                final_pred *= 1.005 # +0.5% per month compounded
            elif name == 'Pessimistic':
                final_pred *= 0.995 # -0.5% per month compounded

            # Update dataframe with Prediction for recursive step
            idx = len(current_df) - 1
            current_df.at[idx, 'Regional_Price'] = final_pred
            if 'National_Price' in current_df.columns:
                 current_df.at[idx, 'National_Price'] = final_pred + current_spread
            
            scenario_dates.append(next_date.strftime("%Y-%m-%d"))
            scenario_prices.append(float(final_pred))
            
        results[name] = {'dates': scenario_dates, 'prices': scenario_prices}

    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commodities = ['cinnamon', 'clove']
    for com in commodities:
        data_path = os.path.join(base_dir, 'data', 'processed', f'{com}_prices.csv')
        if os.path.exists(data_path):
            print(f"Training {com}...")
            df = load_and_prepare_data(data_path)
            train_model(df, commodity=com, epochs=30)
            print(f"Done.")
