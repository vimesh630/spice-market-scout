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
        
        # Add dummy mock features required by model
        np.random.seed(42) # For reproducibility
        df_melted['Region'] = 'Colombo' # Default region
        df_melted['Is_Active_Region'] = 1
        
        # Mock National Price
        df_melted['National_Price'] = df_melted['Regional_Price'] * 1.1 + np.random.normal(0, 50, len(df_melted))
        
        # Add random dummy values for external factors
        external_features = [
            'Seasonal_Impact', 'Local_Production_Volume', 'Local_Export_Volume', 
            'Global_Production_Volume', 'Global_Consumption_Volume', 'Temperature', 
            'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price'
        ]
        
        for col in external_features:
            df_melted[col] = np.random.uniform(10, 100, size=len(df_melted))
            
        df = df_melted

    elif 'Regional_Price' in df.columns and 'Region' in df.columns:
        logger.info("Detected long schema. Enriching with derived features...")
        # Already long format, but likely needs enrichment of external features if missing
        
        np.random.seed(42)
        if 'Is_Active_Region' not in df.columns:
            df['Is_Active_Region'] = 1
        
        if 'National_Price' not in df.columns:
             df['National_Price'] = df['Regional_Price'] * 1.1 + np.random.normal(0, 50, len(df))
             
        external_features = [
            'Seasonal_Impact', 'Local_Production_Volume', 'Local_Export_Volume', 
            'Global_Production_Volume', 'Global_Consumption_Volume', 'Temperature', 
            'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price'
        ]
        
        for col in external_features:
            if col not in df.columns:
                df[col] = np.random.uniform(10, 100, size=len(df))

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

    # Train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )

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


def forecast_multistep(model, df, steps=6, commodity='cinnamon'):
    """
    Iteratively forecast with sophisticated feature simulation and safeguards.
    Ports the advanced logic from LSTM_Clove.ipynb (Cells 17-19) for Clove,
    and provides a stable fallback/adapted version for Cinnamon.
    """
    logger.info(f"Generating {steps}-step forecast for {commodity}...")
    
    # 1. Setup and Statistics
    # Need to determine Grade/Region from df or assume single series
    # The API filters df to a single Grade/Region before calling this.
    # We'll calculate stats from the tail of the dataframe.
    
    current_df = df.copy()
    if 'Date' in current_df.columns:
        current_df = current_df.sort_values('Date')
        
    if len(current_df) < SEQUENCE_LENGTH:
        logger.warning("Not enough data for sequence. Returning empty forecast.")
        return [], []

    last_row = current_df.iloc[-1]
    last_date = last_row['Date'] if 'Date' in last_row else datetime.now()
    history_len = len(current_df)
    
    # Historical stats for variation
    recent_prices = current_df['Regional_Price'].tail(12).values
    historical_std = np.std(recent_prices) if len(recent_prices) > 1 else 50.0
    
    # Calculate averages for production/export if columns exist
    # If not, use defaults from notebook logic
    harvest_production = 2000
    if 'Local_Production_Volume' in current_df.columns and 'Seasonal_Impact' in current_df.columns:
        prod_mean = current_df[current_df['Seasonal_Impact'] == 1]['Local_Production_Volume'].mean()
        if not np.isnan(prod_mean) and prod_mean > 0:
            harvest_production = prod_mean
            
    harvest_export = 200
    if 'Local_Export_Volume' in current_df.columns and 'Seasonal_Impact' in current_df.columns:
        exp_mean = current_df[current_df['Seasonal_Impact'] == 1]['Local_Export_Volume'].mean()
        if not np.isnan(exp_mean) and exp_mean > 0:
            harvest_export = exp_mean
            
    off_season_production = 0
    
    # Initialize current values for random walks
    curr_values = {}
    for col in ['Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD', 
                'National_Price', 'Exchange_Rate', 'Fuel_Price']:
        curr_values[col] = last_row[col] if col in last_row else 0
        
    future_rows = []
    
    # Generate future dates
    future_dates_list = []
    for i in range(1, steps + 1):
        future_dates_list.append(last_date + pd.Timedelta(days=30 * i))
        
    # 2. Simulate Future Environment (Features)
    for i, future_date in enumerate(future_dates_list):
        row = last_row.copy() # Start with last knowns
        
        # Update Time
        row['Date'] = future_date
        if 'Month' in row: row['Month'] = future_date
        row['Year'] = future_date.year
        row['Month_num'] = future_date.month
        row['Quarter'] = future_date.quarter
        
        month = future_date.month
        
        # --- Seasonality Logic ---
        # Clove: Harvest Dec-Feb (12, 1, 2)
        # Cinnamon: Harvest May-Aug (5-8) and Nov-Jan (11, 12, 1)
        
        is_harvest = False
        is_pre_harvest = False
        is_monsoon = False
        is_dry = False
        
        if commodity.lower() == 'clove':
            if month in [12, 1, 2]: is_harvest = True
            if month == 11: is_pre_harvest = True
            if month in [5, 6, 10, 11]: is_monsoon = True
            if month in [12, 1, 2]: is_dry = True
        else: 
            # Cinnamon (approximate)
            if month in [5, 6, 7, 8, 11, 12, 1]: is_harvest = True
            if month in [4, 10]: is_pre_harvest = True
            if month in [5, 6, 10, 11]: is_monsoon = True # SW and NE Monsoons
            if month in [2, 3]: is_dry = True
            
        row['Seasonal_Impact'] = 1 if is_harvest else 0
        
        # --- Production/Export Volumes ---
        if is_harvest:
            row['Local_Production_Volume'] = harvest_production * np.random.uniform(0.8, 1.2)
            row['Local_Export_Volume'] = harvest_export * np.random.uniform(0.8, 1.2)
        elif is_pre_harvest:
            row['Local_Production_Volume'] = harvest_production * 0.7
            row['Local_Export_Volume'] = harvest_export * 0.5
        else:
            row['Local_Production_Volume'] = off_season_production
            row['Local_Export_Volume'] = harvest_export * np.random.uniform(0.1, 0.3)
            
        # Global Production
        base_global = 15000 if is_harvest else 8000
        row['Global_Production_Volume'] = base_global * np.random.uniform(0.7, 1.3)
        
        # --- International Prices & Economics ---
        # Add slight trends and noise
        for col in ['Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD']:
            if col in curr_values:
                # Cycle + Trend + Noise
                cycle = 1 + 0.05 * np.sin(2 * np.pi * month / 12)
                curr_values[col] *= 1.002 * cycle * np.random.uniform(0.97, 1.03)
                row[col] = curr_values[col]
                
        # Exchange Rate (Depreciation trend)
        curr_values['Exchange_Rate'] *= np.random.uniform(1.001, 1.005)
        row['Exchange_Rate'] = curr_values['Exchange_Rate']
        
        # Fuel (Noise)
        curr_values['Fuel_Price'] *= np.random.uniform(0.99, 1.01)
        row['Fuel_Price'] = curr_values['Fuel_Price']
        
        # Inflation (Restricted random walk)
        if 'Inflation_Rate' in last_row:
            row['Inflation_Rate'] = max(-5, min(10, last_row['Inflation_Rate'] + np.random.normal(0, 1)))
            
        # --- Weather ---
        # Temperature: Peak Apr-May, Low Dec-Jan
        base_temp = 27
        temp_var = 2 * np.sin(2 * np.pi * (month - 1) / 12)
        row['Temperature'] = base_temp + temp_var + np.random.normal(0, 0.8)
        
        # Rainfall
        if is_monsoon:
            row['Rainfall'] = 200 + np.random.normal(0, 50)
        elif is_dry:
            row['Rainfall'] = 50 + np.random.normal(0, 20)
        else:
            row['Rainfall'] = 100 + np.random.normal(0, 30)
        row['Rainfall'] = max(0, row['Rainfall'])
        
        # --- National Price (Temporary) ---
        # Will be updated after prediction to maintain spread, but need base for lags
        # Apply seasonal pattern + noise
        nat_seasonal = 1 + 0.03 * np.sin(2 * np.pi * month / 12)
        curr_values['National_Price'] *= nat_seasonal * np.random.uniform(0.98, 1.02)
        row['National_Price'] = curr_values['National_Price']
        
        # Placeholder for target
        row['Regional_Price'] = 0 
        
        future_rows.append(row)
        
    future_df = pd.DataFrame(future_rows)
    extended_df = pd.concat([current_df, future_df], ignore_index=True)
    if 'Date' in extended_df.columns:
        extended_df = extended_df.sort_values('Date').reset_index(drop=True)
    
    # 3. Iterative Prediction
    forecast_prices = []
    forecast_dates = []
    
    cols_to_update_stats = ['Regional_Price', 'National_Price', 'Indonesia_Price_in_USD',
                      'Tanzania_Price_in_USD', 'Madagascar_Price_in_USD',
                      'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate']
                      
    # Identify which columns are actually in the dataframe
    cols_to_update = [c for c in cols_to_update_stats if c in extended_df.columns]
    
    # Also need to ensure TRAIN FEATURES are present and updated
    # We'll use the dynamic feature selection from prepare_sequences logic
    # But explicitly, we need to update 'lag' and 'rolling' features manually or via preprocess
    
    for i in range(steps):
        future_idx = history_len + i
        
        # 3a. Update Lags and Rolling Averages
        # Ideally, we call preprocess_data which handles all this cleanly.
        # But preprocess_data might recalculate everything which is slow but safe.
        # Given n=6, it's fine.
        
        # However, preprocess_data might expect 'training_mode=False' to accept existing encoders.
        # AND we need to ensure 'Regional_Price' at future_idx is not 0 for lag calculation of *later* steps.
        # It IS 0 right now. But we predict it step by step.
        
        # For step i, we need features at i.
        # Lags at i depend on i-1, i-3... which are either history or previously predicted.
        # So we MUST predict sequentially.
        
        # Re-calc features for the WHOLE dataframe (simplest to ensure correctness)
        # Note: preprocess_data uses shift().
        # We need values at future_idx to be based on future_idx-1 etc.
        
        # Only re-process if we updated the price in the *previous* iteration
        if i > 0:
             # Extended DF now has valid price at future_idx - 1
             pass
             
        # Feature Engineering Refresh
        # We can implement a lightweight update or just call the main one.
        # Calling main one is safer for consistency.
        extended_df = preprocess_data(extended_df, training_mode=False)
        
        # 3b. Extract Sequence
        # We need the sequence ENDING at future_idx (exclusive of target? No, standard LSTM X is t-seq...t-1 to predict t?)
        # Wait, prepare_sequences: X = i:i+seq, y = i+seq.
        # So to predict at `future_idx`, we need X from `future_idx - seq` to `future_idx`.
        
        seq_start = future_idx - SEQUENCE_LENGTH
        if seq_start < 0: seq_start = 0 # Should not happen if history check passed
        
        # Get slice
        sequence_df = extended_df.iloc[seq_start : future_idx]
        
        # Predict
        try:
            pred_price = forecast_prices_step(model, sequence_df)
        except Exception as e:
            logger.error(f"Forecast failed at step {i}: {e}")
            pred_price = extended_df.iloc[future_idx-1]['Regional_Price'] # Fallback
            
        # 3c. Adjustments (Noise, Seasonality, Clamping)
        
        # Noise
        noise = np.random.normal(0, historical_std * 0.12)
        pred_price += noise
        
        # Seasonal Adjustment on Price
        month = extended_df.iloc[future_idx]['Month_num'] if 'Month_num' in extended_df.columns else 1
        
        seasonal_factor = 1.0
        if commodity.lower() == 'clove':
            if month in [12, 1, 2]: seasonal_factor = np.random.uniform(0.94, 0.97) # Harvest drop
            elif month in [6, 7, 8]: seasonal_factor = np.random.uniform(1.05, 1.09) # Scarcity rise
            else: seasonal_factor = np.random.uniform(0.99, 1.01)
        else: # Cinnamon
             if month in [5, 6, 7]: seasonal_factor = np.random.uniform(0.95, 0.98)
             elif month in [11, 12]: seasonal_factor = np.random.uniform(0.95, 0.98)
             else: seasonal_factor = np.random.uniform(0.99, 1.01)
             
        pred_price *= seasonal_factor
        
        # Clamping (Max change from recent forecasts/history)
        # Look at last 3 points (history + forecast)
        window = 3
        if i > 0:
            recent = forecast_prices[-min(window, len(forecast_prices)):]
        else:
            recent = [extended_df.iloc[future_idx-1]['Regional_Price']]
            
        recent_avg = np.mean(recent)
        max_change = 0.12 # 12% limit
        pred_price = np.clip(pred_price, recent_avg * (1 - max_change), recent_avg * (1 + max_change))
        
        # 3d. Update DataFrame
        extended_df.at[future_idx, 'Regional_Price'] = pred_price
        
        # Update National Price (Spread Preservation)
        # Calculate percent change of Regional and apply to National with some damping/noise
        if i > 0:
            last_p = extended_df.at[future_idx-1, 'Regional_Price']
            last_n = extended_df.at[future_idx-1, 'National_Price']
            if last_p > 0:
                # Notebook logic: national_change = 0.65 * regional_change + ...
                regional_change_ratio = pred_price / last_p
                # Damped follow
                national_change_ratio = 0.65 * regional_change_ratio + 0.35 * np.random.uniform(0.98, 1.02)
                extended_df.at[future_idx, 'National_Price'] = last_n * national_change_ratio
        
        forecast_prices.append(pred_price)
        date_str = extended_df.iloc[future_idx]['Date'].strftime("%Y-%m-%d")
        forecast_dates.append(date_str)
        
    return forecast_dates, forecast_prices

def forecast_prices_step(model, df_sequence):
    """
    Helper to predict a single step given a pre-prepared feature sequence df.
    """
    # Reconstruct valid features list matching prepare_sequences
    potential_base_features = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 
        'Local_Production_Volume', 'Local_Export_Volume', 
        'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Indonesia_Price_in_USD', 'Madagascar_Price_in_USD', 'Tanzania_Price_in_USD',
        'Year', 'Month_num', 'Quarter'
    ]
    feature_cols = [c for c in potential_base_features if c in df_sequence.columns]
    lag_cols = [col for col in df_sequence.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)
    
    # Ensure sequence length
    if len(df_sequence) != SEQUENCE_LENGTH:
        # Pad if needed (though caller handles this approx)
        # Realistically, should error or pad.
        pass
        
    X_seq = df_sequence[feature_cols].values
    
    # Check shape
    if X_seq.shape[0] < SEQUENCE_LENGTH:
         # Pad with edge
         pad_len = SEQUENCE_LENGTH - X_seq.shape[0]
         # Repeat first row
         first_row = X_seq[0].reshape(1, -1)
         padding = np.repeat(first_row, pad_len, axis=0)
         X_seq = np.vstack([padding, X_seq])
         
    elif X_seq.shape[0] > SEQUENCE_LENGTH:
         X_seq = X_seq[-SEQUENCE_LENGTH:]
         
    # Scale
    X_seq_flat = X_seq.reshape(-1, len(feature_cols))
    
    # Check if we have global scaler
    if scaler_features is None:
        raise ValueError("Scaler not initialized.")
        
    X_seq_scaled = scaler_features.transform(X_seq_flat)
    X_input = X_seq_scaled.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
    
    # Predict
    pred_scaled = model.predict(X_input, verbose=0)
    pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
    
    return float(pred_price)

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
