import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
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
# MODEL_DIR removed as global constant, will be dynamic

# Global scalers - Note: In a production multi-model env, these should not be global 
# but loaded per request. For this refactor, we'll keep them but reload them on demand.
scaler_features = None
scaler_target = None
label_encoders = {}

def preprocess_data(df, training_mode=True):
    """
    Apply feature engineering with strict encoder handling.
    """
    logger.info(f"Preprocessing data (training_mode={training_mode})...")
    
    # Convert 'Date' column to datetime objects if not already
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date
        df = df.sort_values('Date')

    # Assign Month column for sorting/grouping if needed, though notebook used 'Month' as the date column
    # The processed spice_prices.csv has 'Date'. The notebook used 'Month' as the primary date column.
    # We will ensure both exist or map accordingly.
    if 'Month' not in df.columns and 'Date' in df.columns:
        df['Month'] = df['Date'] # Notebook logic uses 'Month' heavily
    
    # Encode categorical variables: Grade, Region
    # Note: 'Is_Active_Region' is numerical boolean, so no encoding needed usually, but check notebook
    for col in ['Grade', 'Region']:
        if col in df.columns:
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
            
            if training_mode:
                # TRAIN: Fit and transform
                df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # INFERENCE: Transform only (handle unseen labels gracefully-ish)
                # If the encoder isn't fitted, we can't transform. Should not happen if trained.
                try:
                    df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Fallback for unseen labels: assign -1 or 0
                    logger.warning(f"Unseen label encountered in {col} during inference. Assigning 0.")
                    df[f'{col}_encoded'] = 0 

    # Create additional time-based features
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month_num'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Ensure 'Month' column is treated as Date (legacy support)
        # If 'Month' exists and is just 1-12, this overwrites it with full Date which is safely what the model expects if it uses 'Month' key.
        # But to be safe, let's keep Month as is if it's already used as feature, OR overwrite if it's main time index.
        # The notebook used 'Month' as the time index.
        df['Month'] = df['Date']

    # Create lag features and rolling averages
    # Need to sort by Grade, Region, Month first
    sort_cols = [c for c in ['Grade', 'Region', 'Month'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    lag_columns = ['Regional_Price', 'National_Price', 'Temperature', 'Rainfall']
    for col in lag_columns:
        if col in df.columns:
            # Need to group by Grade and Region if they exist
            groups = [c for c in ['Grade', 'Region'] if c in df.columns]
            
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

    # Fill NaNs created by lags
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    logger.info(f"Data preprocessed. Shape: {df.shape}")
    return df

def load_and_prepare_data(data_path):
    """
    Loads and prepares the time series data.
    Handles raw CSV schema by melting and adding dummy features if needed.
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if this is the simple schema (Date, Grade_ALBA, Grade_C5, Sentiment)
    if 'Cinnamon_Grade_ALBA' in df.columns:
        logger.info("Detected wide schema. Transforming to long format and enriching...")
        
        # Melt to get Grade and Price
        df_melted = df.melt(id_vars=['Date', 'Market_Sentiment'], 
                           value_vars=['Cinnamon_Grade_ALBA', 'Cinnamon_Grade_C5'],
                           var_name='Grade', value_name='Regional_Price')
        
        # Clean Grade names
        df_melted['Grade'] = df_melted['Grade'].apply(lambda x: x.replace('Cinnamon_Grade_', ''))
        
        # Add dummy features required by model (STATIC DEFAULTS - NO RANDOMNESS)
        df_melted['Region'] = 'Colombo' # Default region
        df_melted['Is_Active_Region'] = 1
        
        # Impute National Price: Fixed 10% spread if missing
        df_melted['National_Price'] = df_melted['Regional_Price'] * 1.1
        
        # Add static default values for external factors (mean imputation equivalent)
        external_features = [
            'Seasonal_Impact', 'Local_Production_Volume', 'Local_Export_Volume', 
            'Global_Production_Volume', 'Global_Consumption_Volume', 'Temperature', 
            'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price'
        ]
        
        for col in external_features:
            df_melted[col] = 0.0 # Placeholder: Use 0 or mean, but valid float
            
        df = df_melted

    elif 'Regional_Price' in df.columns and 'Region' in df.columns:
        logger.info("Detected long schema. Enriching with derived features...")
        # Already long format, but likely needs enrichment of external features if missing
        
        if 'Is_Active_Region' not in df.columns:
            df['Is_Active_Region'] = 1
        
        if 'National_Price' not in df.columns:
             df['National_Price'] = df['Regional_Price'] * 1.1
             
        external_features = [
            'Seasonal_Impact', 'Local_Production_Volume', 'Local_Export_Volume', 
            'Global_Production_Volume', 'Global_Consumption_Volume', 'Temperature', 
            'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price'
        ]
        
        for col in external_features:
            if col not in df.columns:
                df[col] = 0.0 # Static default

    return preprocess_data(df)



def prepare_sequences(df, sequence_length=12, target_col='Regional_Price'):
    """
    Create sequences for LSTM training.
    
    Args:
        df (pd.DataFrame): DataFrame with data.
        sequence_length (int): Length of input sequences.
        target_col (str): Target column name.
        
    Returns:
        tuple: (X_sequences, y_sequences, metadata)
    """
    # Dynamic feature selection: Use all numeric columns except target and metadata
    # This automatically adapts to Clove vs Cinnamon features
    exclude_cols = [target_col, 'Date', 'Month', 'Year', 'Quarter', 'Region', 'Grade', 'Market_Sentiment', 'Month_num']
    # But we explicitly want encoded columns and others
    
    # Better strategy: Start with known features + lags + rolling
    # Base numeric features common to most or specific to one
    potential_base_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD', # Clove specific
        'Year', 'Month_num', 'Quarter'
    ]
    
    feature_cols = [c for c in potential_base_features if c in df.columns]

    # Add lag and rolling features if they exist
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)
    
    valid_feature_cols = feature_cols # Already filtered above
    
    logger.info(f"Using {len(valid_feature_cols)} features: {valid_feature_cols}")

    # Instead of dropping all NaNs, fill them
    df_clean = df.copy()
    
    # Fill numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)

    X_sequences, y_sequences, metadata = [], [], []

    # Handle multiple series (Grade/Region pairs) if present, otherwise treat as single series
    if 'Grade' in df_clean.columns and 'Region' in df_clean.columns:
        for grade in df_clean['Grade'].unique():
            for region in df_clean['Region'].unique():
                subset = df_clean[(df_clean['Grade'] == grade) & (df_clean['Region'] == region)].sort_values('Month') # Assuming 'Month' acts as time index here or use 'Date' if available
                if 'Date' in subset.columns:
                    subset = subset.sort_values('Date')
                
                if len(subset) < sequence_length + 1:
                    continue

                for i in range(len(subset) - sequence_length):
                    X_seq = subset.iloc[i:i + sequence_length][valid_feature_cols].values
                    y_seq = subset.iloc[i + sequence_length][target_col]

                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
                    metadata.append({
                        'grade': grade,
                        'region': region,
                        'date': subset.iloc[i + sequence_length]['Date'] if 'Date' in subset.columns else None
                    })
    else:
        # Simple case: single time series
        if len(df_clean) >= sequence_length + 1:
            for i in range(len(df_clean) - sequence_length):
                X_seq = df_clean.iloc[i:i + sequence_length][valid_feature_cols].values
                y_seq = df_clean.iloc[i + sequence_length][target_col]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                metadata.append({'index': i})

    logger.info(f"Total sequences created: {len(X_sequences)}")
    return np.array(X_sequences), np.array(y_sequences), metadata

def build_lstm_model_tunable(units1=128, units2=64, dropout1=0.2, dropout2=0.2, 
                            dense_units=32, optimizer='adam', learning_rate=0.001, 
                            layer_type='LSTM', use_batch_norm=False, input_shape=None):
    """Build tunable LSTM model with various hyperparameters"""
    model = Sequential()
    
    # Choose layer type
    if layer_type == 'LSTM':
        model.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
    elif layer_type == 'GRU':
        model.add(GRU(units1, return_sequences=True, input_shape=input_shape))
    else:  # SimpleRNN
        model.add(SimpleRNN(units1, return_sequences=True, input_shape=input_shape))
    
    if use_batch_norm:
        model.add(BatchNormalization())
    
    model.add(Dropout(dropout1))
    
    # Second RNN layer
    if layer_type == 'LSTM':
        model.add(LSTM(units2, return_sequences=False))
    elif layer_type == 'GRU':
        model.add(GRU(units2, return_sequences=False))
    else:  # SimpleRNN
        model.add(SimpleRNN(units2, return_sequences=False))
    
    if use_batch_norm:
        model.add(BatchNormalization())
        
    model.add(Dropout(dropout2))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    
    # Configure optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:  # SGD
        opt = SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

class HyperparameterTuner:
    """Hyperparameter tuning class using multiple strategies"""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.best_params = None
        self.best_score = float('inf')
        self.tuning_results = []
    
    def grid_search_tuning(self, param_grid=None, max_trials=20):
        """Grid search hyperparameter tuning"""
        logger.info("Starting Grid Search Hyperparameter Tuning...")
        
        if param_grid is None:
            param_grid = {
                'units1': [64, 128, 256, 512, 1024],
                'units2': [32, 64, 128, 256, 512],
                'dropout1': [0.1, 0.2, 0.3, 0.4, 0.5],
                'dropout2': [0.1, 0.2, 0.3, 0.4, 0.5],
                'dense_units': [16, 32, 64, 128, 256],
                'learning_rate': [0.001, 0.0005, 0.002, 0.005, 0.01],
                'layer_type': ['LSTM', 'GRU'],
                'use_batch_norm': [True, False]
            }
        
        # Generate all combinations and sample randomly if too many
        param_combinations = list(product(*param_grid.values()))
        if len(param_combinations) > max_trials:
            indices = np.random.choice(len(param_combinations), size=max_trials, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        best_val_loss = float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            
            try:
                # Build and train model
                model = build_lstm_model_tunable(**param_dict, input_shape=self.input_shape)
                
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=30,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
                    ]
                )
                
                val_loss = min(history.history['val_loss'])
                
                result = {
                    'trial': i+1,
                    'params': param_dict.copy(),
                    'val_loss': val_loss,
                    'val_mae': min(history.history['val_mae'])
                }
                
                self.tuning_results.append(result)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = param_dict.copy()
                
                # Clean up
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_val_loss
        
        return best_params, best_val_loss
    
    def optuna_tuning(self, n_trials=50):
        """Optuna-based hyperparameter tuning"""
        logger.info("Starting Optuna Hyperparameter Tuning...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'units1': trial.suggest_categorical('units1', [64, 128, 256, 512]),
                'units2': trial.suggest_categorical('units2', [32, 64, 128, 256]),
                'dropout1': trial.suggest_float('dropout1', 0.1, 0.5, step=0.1),
                'dropout2': trial.suggest_float('dropout2', 0.1, 0.5, step=0.1),
                'dense_units': trial.suggest_categorical('dense_units', [16, 32, 64, 128]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'layer_type': trial.suggest_categorical('layer_type', ['LSTM', 'GRU']),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
            }
            
            try:
                model = build_lstm_model_tunable(**params, input_shape=self.input_shape)
                
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=25,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
                    ]
                )
                
                val_loss = min(history.history['val_loss'])
                
                # Clean up
                del model
                tf.keras.backend.clear_session()
                
                return val_loss
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return study.best_params, study.best_value
    
    def random_search_tuning(self, n_trials=30):
        """Random search hyperparameter tuning"""
        logger.info("Starting Random Search Hyperparameter Tuning...")
        
        best_val_loss = float('inf')
        best_params = None
        
        for i in range(n_trials):
            # Randomly sample hyperparameters
            params = {
                'units1': np.random.choice([64, 128, 256, 512]),
                'units2': np.random.choice([32, 64, 128, 256]),
                'dropout1': np.random.uniform(0.1, 0.5),
                'dropout2': np.random.uniform(0.1, 0.5),
                'dense_units': np.random.choice([16, 32, 64, 128]),
                'learning_rate': 10 ** np.random.uniform(-4, -2), # Log uniform
                'layer_type': np.random.choice(['LSTM', 'GRU']),
                'use_batch_norm': np.random.choice([True, False]),
                'optimizer': np.random.choice(['adam', 'rmsprop'])
            }
            
            try:
                model = build_lstm_model_tunable(**params, input_shape=self.input_shape)
                
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=25,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
                    ]
                )
                
                val_loss = min(history.history['val_loss'])
                
                result = {
                    'trial': i+1,
                    'params': params.copy(),
                    'val_loss': val_loss,
                    'val_mae': min(history.history['val_mae'])
                }
                
                self.tuning_results.append(result)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params.copy()
                
                # Clean up
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_val_loss
        
        return best_params, best_val_loss

def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, 
                                 method='optuna', n_trials=30):
    """Main function to perform hyperparameter tuning"""
    print(f"\nStarting Hyperparameter Tuning using {method.upper()} method...")
    
    tuner = HyperparameterTuner(X_train, y_train, X_val, y_val, input_shape)
    
    if method == 'optuna':
        best_params, best_score = tuner.optuna_tuning(n_trials=n_trials)
    elif method == 'random':
        best_params, best_score = tuner.random_search_tuning(n_trials=n_trials)
    elif method == 'grid':
        best_params, best_score = tuner.grid_search_tuning(max_trials=n_trials)
    else:
        raise ValueError("Method must be 'optuna', 'grid', or 'random'")
    
    return best_params, best_score, tuner

def build_lstm_model(input_shape, best_params=None):
    """Build LSTM model with optional best parameters from tuning"""
    if best_params is None:
        # Default parameters
        best_params = {
            'units1': 128,
            'units2': 64,
            'dropout1': 0.2,
            'dropout2': 0.2,
            'dense_units': 32,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'layer_type': 'LSTM',
            'use_batch_norm': False
        }
    
    return build_lstm_model_tunable(**best_params, input_shape=input_shape)

def save_model(model, history, results, model_dir):
    """
    Saves the model, history, and results.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'lstm_model.keras')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save history
    history_path = os.path.join(model_dir, 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save results
    results_path = os.path.join(model_dir, 'results.json')
    # Convert numpy types to python types for JSON serialization
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.float64): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, default=convert, indent=4)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results JSON: {e}")

    # Save scalers - MUST save specific to this model
    with open(os.path.join(model_dir, 'scaler_features.pkl'), 'wb') as f:
        pickle.dump(scaler_features, f)
    with open(os.path.join(model_dir, 'scaler_target.pkl'), 'wb') as f:
        pickle.dump(scaler_target, f)
        
    # Save Label Encoders
    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature list to know what features were used
    if 'feature_cols' in results:
         with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
            json.dump(results['feature_cols'], f)

def train_model(df, commodity='cinnamon', use_tuning=True, tuning_method='optuna', n_tuning_trials=20, epochs=100, batch_size=32):
    """Train the forecasting model with optional hyperparameter tuning"""
    global scaler_features, scaler_target
    
    # Re-initialize scalers for new training to avoid contamination
    scaler_features = StandardScaler() # Use StandardScaler for features as per notebook improvement
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    logger.info("Preparing sequences...")
    X, y, metadata = prepare_sequences(df, SEQUENCE_LENGTH)

    if len(X) == 0:
        raise ValueError("No sequences could be created. Check if there's enough data.")
        
    # Capture feature columns used (hacky way, should return from prepare_sequences)
    # We will reconstruct the list used in prepare_sequences to save it
    potential_base_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD',
        'Year', 'Month_num', 'Quarter'
    ]
    feature_cols = [c for c in potential_base_features if c in df.columns]
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)

    logger.info(f"Created {len(X)} sequences with shape {X.shape}")

    # Scale features and target
    logger.info("Scaling features...")
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled_reshaped = scaler_features.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)

    y_scaled = scaler_target.fit_transform(y.reshape(-1, 1)).flatten()

    # Chronological Split (No Shuffling)
    # Train: 80% | Test: 20%  (Within Train -> Train: 80% | Val: 20%)
    
    # 1. Total Split (Train vs Test)
    split_idx_test = int(len(X_scaled) * 0.8)
    
    X_temp = X_scaled[:split_idx_test]
    y_temp = y_scaled[:split_idx_test]
    
    X_test = X_scaled[split_idx_test:]
    y_test = y_scaled[split_idx_test:]
    
    # 2. Train vs Validation Split
    split_idx_val = int(len(X_temp) * 0.8)
    
    X_train = X_temp[:split_idx_val]
    y_train = y_temp[:split_idx_val]
    
    X_val = X_temp[split_idx_val:]
    y_val = y_temp[split_idx_val:]
    
    # Save feature columns strictly
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f) # Using the extended list including lags

    logger.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    best_params = None
    tuner = None
    
    # Hyperparameter tuning
    if use_tuning:
        logger.info(f"Performing hyperparameter tuning using {tuning_method} method...")
        best_params, best_score, tuner = perform_hyperparameter_tuning(
            X_train, y_train, X_val, y_val, input_shape, 
            method=tuning_method, n_trials=n_tuning_trials
        )
        
        logger.info("Hyperparameter Tuning Results:")
        logger.info(f"Best validation loss: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
    else:
        logger.info("Skipping hyperparameter tuning, using default parameters...")

    # Build and train final model with best parameters
    logger.info("Building final model with optimized parameters...")
    model = build_lstm_model(input_shape, best_params)
    
    logger.info("Training final model...")
    # Use longer training for final model
    final_epochs = epochs if epochs else (150 if use_tuning else 100)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=final_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=8, factor=0.5, verbose=1)
        ]
    )

    # Evaluate model on test set
    logger.info("Evaluating final model on test set...")
    y_pred = model.predict(X_test)
    y_pred_unscaled = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    logger.info(f"Final Model Performance on Test Set:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R2: {r2:.4f}")
    
    # Create comprehensive results dictionary
    results = {
        'mae': mae, 
        'rmse': rmse, 
        'r2': r2,
        'best_params': best_params,
        'tuning_method': tuning_method if use_tuning else None,
        'tuning_used': use_tuning,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'feature_cols': feature_cols
    }
    
    # Add tuning results if available
    if tuner and hasattr(tuner, 'tuning_results'):
        results['tuning_results'] = tuner.tuning_results
    
    # Save everything
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    save_model(model, history, results, model_dir)
    
    return model, history, results

def load_artifacts(commodity='cinnamon'):
    """Load model and scalers for inference"""
    global scaler_features, scaler_target
    
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    logger.info(f"Loading artifacts from {model_dir}")
    
    try:
        # Load model
        model_path = os.path.join(model_dir, 'lstm_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = load_model(model_path)

        
        # Load scalers
        with open(os.path.join(model_dir, 'scaler_features.pkl'), 'rb') as f:
            scaler_features = pickle.load(f)
        with open(os.path.join(model_dir, 'scaler_target.pkl'), 'rb') as f:
            scaler_target = pickle.load(f)
            
        # --- NEW: Load Encoders ---
        encoder_path = os.path.join(model_dir, 'label_encoders.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoders = pickle.load(f)
        else:
            logger.warning("Label encoders not found! Inference may be inaccurate.")
            
        logger.info("Artifacts loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return None


def forecast_prices(model, df, days_ahead=30):
    """
    Generate future price forecasts using the trained model.
    """
    # Prepare the most recent sequence for prediction
    # Dynamic feature selection based on available columns and common logic
    # In a real system, we should save and load the exact feature list used in training
    # For now, we reconstruct it similar to prepare_sequences
    potential_base_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD',
        'Year', 'Month_num', 'Quarter'
    ]
    feature_cols = [c for c in potential_base_features if c in df.columns]
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)
    valid_feature_cols = [col for col in feature_cols if col in df.columns]

    # Get the last sequence_length rows
    last_sequence_df = df.iloc[-SEQUENCE_LENGTH:].copy()
    
    if len(last_sequence_df) < SEQUENCE_LENGTH:
        raise ValueError("Not enough data to generate forecast")

    # Assuming we need to forecast iteratively or just one step?
    # For now, let's implement a single next-step forecast for demonstration, 
    # or a multi-step if the prompt implied it. 
    # The user said "forecast_prices" (plural), possibly meaning a series.
    # However, without a multi-step training approach or a recursive strategy implemented, 
    # we can only reliably predict one step unless we assume features for future steps are known.
    # I will implement a single step prediction for now involving the latest data.
    
    # Scale
    X_seq = last_sequence_df[valid_feature_cols].values
    X_seq_reshaped = X_seq.reshape(1, SEQUENCE_LENGTH, len(valid_feature_cols))
    
    # We need to scale this sequence using the same scaler_features used in training.
    # Since scaler_features is global, we can use it, but in production we should load it.
    
    # Important: The scaler expects (n_samples, n_features). 
    # Our scaler was fitted on X_reshaped which was (n_samples * n_timesteps, n_features).
    # So we should flatten, scale, and reshape.
    
    X_seq_flat = X_seq_reshaped.reshape(-1, len(valid_feature_cols))
    X_seq_scaled = scaler_features.transform(X_seq_flat) # Use transform, not fit_transform
    X_input = X_seq_scaled.reshape(1, SEQUENCE_LENGTH, len(valid_feature_cols))
    
    # Predict
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler_target.inverse_transform(predicted_scaled)[0][0]
    
    return float(predicted_price)



def train_all_models(commodity='cinnamon'):
    """
    Train models for the specified commodity.
    This function is called by the update_data.py script.
    """
    logger.info(f"Starting model training (All Models) for {commodity}...")
    
    # Construct data path
    # Assuming data is in data/processed/{commodity}_prices.csv relative to project root
    # BASE_DIR is c:\Vimesh\spice-market-scout
    # Data is in c:\Vimesh\data\processed
    # We need to adjust path from BASE_DIR which is defined as src parent
    _PROJECT_ROOT = BASE_DIR
    data_path = os.path.join(_PROJECT_ROOT, 'data', 'processed', f'{commodity}_prices.csv')
    
    # Check if data exists
    if not os.path.exists(data_path):
        # Fallback for different CWD
        data_path = os.path.join('data', 'processed', f'{commodity}_prices.csv')
        
    if not os.path.exists(data_path):
        # Allow for absolute path from config if needed, or just error out
        logger.error(f"Data file not found at {data_path}")
        return
        
    try:
        # Load and prepare data
        df = load_and_prepare_data(data_path)
        
        # Train model
        train_model(df, commodity=commodity, use_tuning=True, tuning_method='optuna', n_tuning_trials=10)
        
        logger.info(f"Successfully trained models for {commodity}")
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise e


def forecast_multistep(model, df, steps=24, commodity='cinnamon'):
    """
    Stabilized 24-month forecast using Additive Logic (No Multipliers).
    Strictly uses feature_cols.json and Calendar-Aware dates.
    """
    # --- SANITY CHECK PRINT ---
    print("\n" + "="*50)
    print("--- 🚀 RUNNING STABILIZED FORECAST V2 (PHASE 1 REFACTOR) ---")
    print("="*50 + "\n")
    
    logger.info(f"Generating {steps}-step stabilized forecast for {commodity}...")
    
    # --- 1. FEATURE CONSISTENCY CHECK ---
    model_dir = os.path.join(BASE_DIR, 'models', f'lstm_{commodity}')
    feature_cols_path = os.path.join(model_dir, 'feature_cols.json')
    
    if not os.path.exists(feature_cols_path):
        logger.error(f"CRITICAL: feature_cols.json not found at {feature_cols_path}")
        # We must fail as per "Strict" requirements
        raise FileNotFoundError("feature_cols.json missing. Please retrain the model to generate this file.")
        
    with open(feature_cols_path, 'r') as f:
        required_features = json.load(f)
        
    logger.info(f"Loaded {len(required_features)} strict features from schema.")

    current_df = df.copy()
    if 'Date' in current_df.columns:
        current_df = current_df.sort_values('Date')
        
    # 2. Calculate the static "Spread" (National Price - Regional Price)
    last_row = current_df.iloc[-1]
    current_spread = 0
    if 'National_Price' in last_row and 'Regional_Price' in last_row:
        current_spread = last_row['National_Price'] - last_row['Regional_Price']

    future_dates = []
    future_prices = []
    
    # Base date for offset calculation
    base_date = last_row['Date']

    for i in range(1, steps + 1):
        # A. Setup Next Date (Calendar-Aware)
        # Using DateOffset ensures we land on the same day next month (or end of month)
        next_date = base_date + pd.DateOffset(months=i)
        
        # B. Naive Forecast: Copy last known features (Freeze external factors)
        next_row = current_df.iloc[-1].copy()
        next_row['Date'] = next_date
        next_row['Month'] = next_date
        next_row['Year'] = next_date.year
        next_row['Month_num'] = next_date.month
        next_row['Quarter'] = next_date.quarter
        
        # Append temporarily to calculate lags
        next_row_df = pd.DataFrame([next_row])
        current_df = pd.concat([current_df, next_row_df], ignore_index=True)
        
        # C. Update Lags/Rolling features based on history
        current_df = preprocess_data(current_df, training_mode=False)
        
        # D. Predict Next Price with STRICT FEATURES
        # Ensure we only use the columns in required_features
        # And ensure they exist
        missing_cols = [c for c in required_features if c not in current_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
            
        # Extract sequence and filter columns
        input_sequence = current_df.iloc[-SEQUENCE_LENGTH:][required_features]
        
        try:
            # Inline checks and prediction for strict compliance
            X_seq = input_sequence.values
            
            # Rescale
            X_seq_flat = X_seq.reshape(-1, len(required_features))
            X_seq_scaled = scaler_features.transform(X_seq_flat)
            X_input = X_seq_scaled.reshape(1, SEQUENCE_LENGTH, len(required_features))
            
            pred_scaled = model.predict(X_input, verbose=0)
            pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
            
        except Exception as e:
            logger.error(f"Prediction failed at step {i}: {e}")
            pred_price = current_df.iloc[-2]['Regional_Price']

        # E. STABILIZATION
        prev_price = current_df.iloc[-2]['Regional_Price']
        max_change = prev_price * 0.10
        pred_price = np.clip(pred_price, prev_price - max_change, prev_price + max_change)
        
        # Update DataFrame
        idx = len(current_df) - 1
        current_df.at[idx, 'Regional_Price'] = pred_price
        
        if 'National_Price' in current_df.columns:
            current_df.at[idx, 'National_Price'] = pred_price + current_spread

        future_dates.append(next_date.strftime("%Y-%m-%d"))
        future_prices.append(float(pred_price))

    return future_dates, future_prices


if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    commodities = ['cinnamon', 'clove']
    
    for com in commodities:
        data_path = os.path.join(base_dir, 'data', 'processed', f'{com}_prices.csv')
        
        if os.path.exists(data_path):
            print(f"--- Training {com.upper()} ---")
            print(f"Loading data from {data_path}")
            try:
                df = load_and_prepare_data(data_path)
                
                # Run training
                train_model(df, commodity=com, use_tuning=False, epochs=15)
                print(f"{com} training complete.")
                
            except Exception as e:
                print(f"{com} training failed: {e}")
                import traceback
                traceback.print_exc()

        else:
            print(f"Data file not found at {data_path}")
