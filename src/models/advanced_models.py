"""
Advanced ML Models for Quantitative Finance
Includes LSTM, Transformer, XGBoost, and ARIMA-GARCH models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import joblib
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output

class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x[:, -1, :])
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesPredictor:
    """Base class for time series prediction models."""
    
    def __init__(self, model_type: str = "lstm", config: Dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close', 
                    sequence_length: int = 60, test_size: float = 0.2) -> Tuple:
        """Prepare data for time series prediction."""
        # Select features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Volatility']
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if target_col not in data.columns:
            target_col = 'Close'
        
        # Prepare features and target
        features = data[available_cols].values
        target = data[target_col].values
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, available_cols
    
    def fit(self, data: pd.DataFrame, target_col: str = 'Close', 
            sequence_length: int = 60, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Fit the model."""
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(
            data, target_col, sequence_length
        )
        
        if self.model_type == "lstm":
            return self._fit_lstm(X_train, X_test, y_train, y_test, epochs, batch_size)
        elif self.model_type == "transformer":
            return self._fit_transformer(X_train, X_test, y_train, y_test, epochs, batch_size)
        elif self.model_type == "xgboost":
            return self._fit_xgboost(X_train, X_test, y_train, y_test)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _fit_lstm(self, X_train, X_test, y_train, y_test, epochs: int, batch_size: int) -> Dict:
        """Fit LSTM model."""
        input_size = X_train.shape[2]
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.get('hidden_size', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2)
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('lr', 0.001))
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor).squeeze()
                test_loss = criterion(test_outputs, y_test_tensor)
                
            train_losses.append(epoch_loss / len(train_loader))
            test_losses.append(test_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Test Loss = {test_losses[-1]:.6f}")
        
        self.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
    
    def _fit_transformer(self, X_train, X_test, y_train, y_test, epochs: int, batch_size: int) -> Dict:
        """Fit Transformer model."""
        input_size = X_train.shape[2]
        
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.config.get('d_model', 128),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            dropout=self.config.get('dropout', 0.1)
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('lr', 0.001))
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor).squeeze()
                test_loss = criterion(test_outputs, y_test_tensor)
                
            train_losses.append(epoch_loss / len(train_loader))
            test_losses.append(test_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Test Loss = {test_losses[-1]:.6f}")
        
        self.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
    
    def _fit_xgboost(self, X_train, X_test, y_train, y_test) -> Dict:
        """Fit XGBoost model."""
        # Reshape for XGBoost (flatten sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            random_state=42
        )
        
        self.model.fit(X_train_flat, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_flat)
        y_test_pred = self.model.predict(X_test_flat)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.is_fitted = True
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': self.model.feature_importances_
        }
    
    def predict(self, data: pd.DataFrame, sequence_length: int = 60) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Volatility']
        available_cols = [col for col in feature_cols if col in data.columns]
        features = data[available_cols].values
        features_scaled = self.scaler.transform(features)
        
        if self.model_type in ["lstm", "transformer"]:
            # Create sequences
            X = []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
            X = np.array(X)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor).squeeze().numpy()
            
            # Pad with NaN for the first sequence_length values
            full_predictions = np.full(len(data), np.nan)
            full_predictions[sequence_length:] = predictions
            
            return full_predictions
            
        elif self.model_type == "xgboost":
            # Reshape for XGBoost
            X = features_scaled.reshape(1, -1)
            predictions = self.model.predict(X)
            return predictions
    
    def save_model(self, path: str):
        """Save the model."""
        if self.model_type in ["lstm", "transformer"]:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'config': self.config,
                'model_type': self.model_type
            }, path)
        elif self.model_type == "xgboost":
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'model_type': self.model_type
            }, path)
    
    def load_model(self, path: str):
        """Load the model."""
        if self.model_type in ["lstm", "transformer"]:
            checkpoint = torch.load(path)
            self.config = checkpoint['config']
            self.scaler = checkpoint['scaler']
            
            if self.model_type == "lstm":
                self.model = LSTMModel(
                    input_size=self.config.get('input_size', 8),
                    hidden_size=self.config.get('hidden_size', 128),
                    num_layers=self.config.get('num_layers', 2),
                    dropout=self.config.get('dropout', 0.2)
                )
            else:
                self.model = TransformerModel(
                    input_size=self.config.get('input_size', 8),
                    d_model=self.config.get('d_model', 128),
                    nhead=self.config.get('nhead', 8),
                    num_layers=self.config.get('num_layers', 6),
                    dropout=self.config.get('dropout', 0.1)
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        elif self.model_type == "xgboost":
            checkpoint = joblib.load(path)
            self.model = checkpoint['model']
            self.scaler = checkpoint['scaler']
            self.config = checkpoint['config']
        
        self.is_fitted = True

class ARIMAGARCHModel:
    """ARIMA-GARCH model for volatility forecasting."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.arima_model = None
        self.garch_model = None
        self.is_fitted = False
        
    def fit(self, data: pd.Series, arima_order: Tuple = (1, 1, 1), 
            garch_order: Tuple = (1, 1)) -> Dict:
        """Fit ARIMA-GARCH model."""
        try:
            # Check for stationarity
            if not self._is_stationary(data):
                logger.info("Data is not stationary, differencing applied")
                data = data.diff().dropna()
            
            # Fit ARIMA model
            self.arima_model = ARIMA(data, order=arima_order)
            self.arima_fitted = self.arima_model.fit()
            
            # Get residuals for GARCH
            residuals = self.arima_fitted.resid
            
            # Fit GARCH model
            self.garch_model = arch_model(residuals, vol='GARCH', p=garch_order[0], q=garch_order[1])
            self.garch_fitted = self.garch_model.fit(disp='off')
            
            self.is_fitted = True
            
            return {
                'arima_aic': self.arima_fitted.aic,
                'arima_bic': self.arima_fitted.bic,
                'garch_aic': self.garch_fitted.aic,
                'garch_bic': self.garch_fitted.bic,
                'arima_summary': self.arima_fitted.summary(),
                'garch_summary': self.garch_fitted.summary()
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA-GARCH model: {e}")
            return {}
    
    def _is_stationary(self, data: pd.Series) -> bool:
        """Check if time series is stationary."""
        result = adfuller(data.dropna())
        return result[1] < 0.05
    
    def predict(self, steps: int = 30) -> Dict:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # ARIMA forecast
            arima_forecast = self.arima_fitted.forecast(steps=steps)
            
            # GARCH volatility forecast
            garch_forecast = self.garch_fitted.forecast(horizon=steps)
            volatility_forecast = np.sqrt(garch_forecast.variance.values[-1, :])
            
            return {
                'mean_forecast': arima_forecast,
                'volatility_forecast': volatility_forecast,
                'confidence_intervals': {
                    'lower': arima_forecast - 1.96 * volatility_forecast,
                    'upper': arima_forecast + 1.96 * volatility_forecast
                }
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}
    
    def save_model(self, path: str):
        """Save the model."""
        if self.is_fitted:
            joblib.dump({
                'arima_fitted': self.arima_fitted,
                'garch_fitted': self.garch_fitted,
                'config': self.config
            }, path)
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = joblib.load(path)
        self.arima_fitted = checkpoint['arima_fitted']
        self.garch_fitted = checkpoint['garch_fitted']
        self.config = checkpoint['config']
        self.is_fitted = True

class EnsembleModel:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models: List[str] = None, weights: List[float] = None):
        self.models = models or ["lstm", "transformer", "xgboost"]
        self.weights = weights or [0.4, 0.4, 0.2]  # Equal weights by default
        self.fitted_models = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Fit all models in the ensemble."""
        results = {}
        
        for model_name in self.models:
            logger.info(f"Fitting {model_name} model...")
            
            if model_name in ["lstm", "transformer", "xgboost"]:
                model = TimeSeriesPredictor(model_name)
                result = model.fit(data, target_col)
                self.fitted_models[model_name] = model
                results[model_name] = result
                
            elif model_name == "arima_garch":
                model = ARIMAGARCHModel()
                result = model.fit(data[target_col])
                self.fitted_models[model_name] = model
                results[model_name] = result
        
        self.is_fitted = True
        return results
    
    def predict(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before making predictions")
        
        predictions = {}
        ensemble_pred = None
        
        for model_name, model in self.fitted_models.items():
            if model_name in ["lstm", "transformer", "xgboost"]:
                pred = model.predict(data)
                predictions[model_name] = pred
                
                if ensemble_pred is None:
                    ensemble_pred = self.weights[self.models.index(model_name)] * pred
                else:
                    ensemble_pred += self.weights[self.models.index(model_name)] * pred
                    
            elif model_name == "arima_garch":
                pred_result = model.predict(steps=len(data))
                if pred_result:
                    pred = pred_result['mean_forecast']
                    predictions[model_name] = pred
                    
                    if ensemble_pred is None:
                        ensemble_pred = self.weights[self.models.index(model_name)] * pred
                    else:
                        ensemble_pred += self.weights[self.models.index(model_name)] * pred
        
        predictions['ensemble'] = ensemble_pred
        return predictions
    
    def save_models(self, base_path: str):
        """Save all models."""
        for model_name, model in self.fitted_models.items():
            path = f"{base_path}_{model_name}.pkl"
            model.save_model(path)
    
    def load_models(self, base_path: str):
        """Load all models."""
        for model_name in self.models:
            path = f"{base_path}_{model_name}.pkl"
            if os.path.exists(path):
                if model_name in ["lstm", "transformer", "xgboost"]:
                    model = TimeSeriesPredictor(model_name)
                elif model_name == "arima_garch":
                    model = ARIMAGARCHModel()
                
                model.load_model(path)
                self.fitted_models[model_name] = model
        
        self.is_fitted = len(self.fitted_models) > 0 