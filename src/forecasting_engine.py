import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import optuna
import pickle
import json
from itertools import product
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
SEQUENCE_LENGTH = 12
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'lstm_cinnamon')


# Global scalers (will be initialized in training)
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
label_encoders = {}

def preprocess_data(df):
    """
    Apply feature engineering, encoding and handling missing values.
    """
    logger.info("Preprocessing data...")
    
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
            # Handle unknown values in future/inference by extending fit? For now, fit_transform or transform
            # For simplicity in this script, we fit_transform. In production, load saved encoders.
            try:
                df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            except Exception as e:
                logger.warning(f"Could not encode {col}: {e}")

    # Create additional time-based features
    if 'Month' in df.columns:
        df['Year'] = df['Month'].dt.year
        df['Month_num'] = df['Month'].dt.month
        df['Quarter'] = df['Month'].dt.quarter

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
        # In a real scenario, these would come from external sources
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
            # Generate somewhat realistic looking random series
            df_melted[col] = np.random.uniform(10, 100, size=len(df_melted))
            
        df = df_melted
        
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
    feature_cols = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 'Local_Production_Volume',
        'Local_Export_Volume', 'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Year', 'Month_num', 'Quarter'
    ]

    # Add lag and rolling features if they exist
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    feature_cols.extend(lag_cols)
    
    # Filter only columns that actually exist in df
    valid_feature_cols = [col for col in feature_cols if col in df.columns]

    # Instead of dropping all NaNs, fill them
    df_clean = df.copy()
    df_clean = df_clean.fillna(method='bfill').fillna(method='ffill')

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

    # Save scalers
    with open(os.path.join(model_dir, 'scaler_features.pkl'), 'wb') as f:
        pickle.dump(scaler_features, f)
    with open(os.path.join(model_dir, 'scaler_target.pkl'), 'wb') as f:
        pickle.dump(scaler_target, f)

def train_model(df, use_tuning=True, tuning_method='optuna', n_tuning_trials=20, epochs=100, batch_size=32):
    """Train the forecasting model with optional hyperparameter tuning"""
    global scaler_features, scaler_target
    
    logger.info("Preparing sequences...")
    X, y, metadata = prepare_sequences(df, SEQUENCE_LENGTH)

    if len(X) == 0:
        raise ValueError("No sequences could be created. Check if there's enough data.")

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
        'final_val_loss': history.history['val_loss'][-1]
    }
    
    # Add tuning results if available
    if tuner and hasattr(tuner, 'tuning_results'):
        results['tuning_results'] = tuner.tuning_results
    
    # Save everything
    save_model(model, history, results, MODEL_DIR)
    
    return model, history, results

def load_artifacts():
    """Load model and scalers for inference"""
    global scaler_features, scaler_target
    
    logger.info(f"Loading artifacts from {MODEL_DIR}")
    
    try:
        # Load model
        model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = load_model(model_path)

        
        # Load scalers
        with open(os.path.join(MODEL_DIR, 'scaler_features.pkl'), 'rb') as f:
            scaler_features = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler_target.pkl'), 'rb') as f:
            scaler_target = pickle.load(f)
            
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
    feature_cols = [
        'Grade_encoded', 'Region_encoded', 'Is_Active_Region',
        'National_Price', 'Seasonal_Impact', 'Local_Production_Volume',
        'Local_Export_Volume', 'Global_Production_Volume', 'Global_Consumption_Volume',
        'Temperature', 'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price',
        'Year', 'Month_num', 'Quarter'
    ]
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


if __name__ == "__main__":
    # Example usage with mock data adapter for verification
    # Since the original complex dataset is likely missing, we adapt the simple mock data
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'processed', 'spice_prices.csv')
    
    if os.path.exists(data_path):
        print(f"Loading mock data for verification from {data_path}")
        raw_df = pd.read_csv(data_path)
        if 'Date' in raw_df.columns:
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        
        # Melt to get Grade and Price if seemingly pivoted
        if 'Cinnamon_Grade_ALBA' in raw_df.columns:
            df_melted = raw_df.melt(id_vars=['Date', 'Market_Sentiment'], 
                                   value_vars=['Cinnamon_Grade_ALBA', 'Cinnamon_Grade_C5'],
                                   var_name='Grade', value_name='Regional_Price')
            
            # Map columns
            df_melted['Grade'] = df_melted['Grade'].apply(lambda x: x.replace('Cinnamon_Grade_', ''))
            
            # Add missing dummy features required by prepare_sequences
            df_melted['Region'] = 'Colombo' # Dummy region
            df_melted['Is_Active_Region'] = 1
            df_melted['National_Price'] = df_melted['Regional_Price'] * 1.1 + np.random.normal(0, 50, len(df_melted))
            
            # Add random dummy values for other features
            for col in ['Seasonal_Impact', 'Local_Production_Volume', 'Local_Export_Volume', 
                       'Global_Production_Volume', 'Global_Consumption_Volume', 'Temperature', 
                       'Rainfall', 'Exchange_Rate', 'Inflation_Rate', 'Fuel_Price']:
                df_melted[col] = np.random.uniform(10, 100, size=len(df_melted))
            
            # Use the new preprocess function
            processed_df = preprocess_data(df_melted)
            
            # Run training (fast mode)
            print("Running training test...")
            train_model(processed_df, use_tuning=False, epochs=2)
            
        else:
            print("Mock data schema not recognized (expected pivoted grades).")
    else:
        print(f"Data file not found at {data_path}")
