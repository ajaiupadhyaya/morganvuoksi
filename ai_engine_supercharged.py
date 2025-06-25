#!/usr/bin/env python3
"""
MorganVuoksi AI Engine - Supercharged ML/AI Trading System
Advanced machine learning pipeline for quantitative finance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    predictions: np.ndarray
    confidence: np.ndarray
    features_importance: Dict[str, float]
    model_metadata: Dict[str, Any]

@dataclass
class MarketSignal:
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    confidence: float  # 0-1
    timeframe: str
    rationale: str
    risk_score: float

class AdvancedLSTMNetwork(nn.Module):
    """Advanced LSTM with attention mechanism for financial time series."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.2, output_size: int = 1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.classifier = nn.Linear(hidden_size // 2, output_size)
        self.confidence_head = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global max pooling
        pooled = torch.max(attended_out, dim=1)[0]
        
        # Feature processing
        features = self.feature_processor(pooled)
        
        # Outputs
        predictions = self.classifier(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return predictions, confidence, attention_weights

class TransformerTimeSeriesModel(nn.Module):
    """Transformer model for financial time series prediction."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.price_head = nn.Linear(d_model, 1)
        self.volatility_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 3)  # up, down, sideways
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        
        # Multiple outputs
        price_pred = self.price_head(pooled)
        vol_pred = torch.exp(self.volatility_head(pooled))  # Ensure positive
        direction_pred = torch.softmax(self.direction_head(pooled), dim=-1)
        
        return price_pred, vol_pred, direction_pred

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ReinforcementLearningAgent(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 256):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage network
        self.advantage_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, state):
        value = self.value_net(state)
        advantage = self.advantage_net(state)
        
        # Dueling architecture
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

class SuperchargedAIEngine:
    """Advanced AI engine for quantitative trading."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.models = {}
        self.model_metadata = {}
        self.feature_columns = []
        
        # Initialize models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all AI models."""
        logger.info("Initializing AI models...")
        
        # Model configurations
        configs = {
            'lstm_advanced': {'input_size': 50, 'hidden_size': 128, 'num_layers': 3},
            'transformer': {'input_size': 50, 'd_model': 128, 'nhead': 8, 'num_layers': 6},
            'rl_agent': {'state_size': 50, 'action_size': 3, 'hidden_size': 256}
        }
        
        # Initialize models
        try:
            self.models['lstm'] = AdvancedLSTMNetwork(**configs['lstm_advanced']).to(self.device)
            self.models['transformer'] = TransformerTimeSeriesModel(**configs['transformer']).to(self.device)
            self.models['rl_agent'] = ReinforcementLearningAgent(**configs['rl_agent']).to(self.device)
            
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare advanced features for AI models."""
        if data.empty:
            return data
        
        features = data.copy()
        
        # Price features
        if 'Close' in features.columns:
            features['returns'] = features['Close'].pct_change()
            features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
            features['cumulative_returns'] = (1 + features['returns']).cumprod()
            
            # Volatility features
            features['realized_volatility'] = features['returns'].rolling(20).std()
            features['garman_klass_vol'] = self._calculate_garman_klass_volatility(features)
            
            # Momentum features
            for period in [5, 10, 20, 50]:
                features[f'momentum_{period}'] = features['Close'] / features['Close'].shift(period) - 1
                features[f'rsi_{period}'] = self._calculate_rsi(features['Close'], period)
            
            # Mean reversion features
            sma_20 = features['Close'].rolling(20).mean()
            features['price_position'] = (features['Close'] - sma_20) / sma_20
            
            # Fractal features
            features['hurst_exponent'] = features['returns'].rolling(50).apply(self._calculate_hurst_exponent)
            
            # Microstructure features
            if 'Volume' in features.columns:
                features['volume_profile'] = features['Volume'] / features['Volume'].rolling(20).mean()
                features['price_volume_correlation'] = features['returns'].rolling(20).corr(features['Volume'].pct_change())
        
        # Cross-asset features (simulated)
        features['market_beta'] = features['returns'].rolling(60).cov(features['returns']) / features['returns'].rolling(60).var()
        
        # Regime features
        features = self._add_regime_features(features)
        
        # Options features (simulated)
        features = self._add_options_features(features)
        
        # Clean features
        features = features.dropna()
        
        # Store feature columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
        
        return features
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        if not all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
            return pd.Series(0, index=data.index)
        
        log_hl = np.log(data['High'] / data['Low'])
        log_co = np.log(data['Close'] / data['Open'])
        
        gk_vol = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return gk_vol.rolling(20).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with improved numerical stability."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / (loss + 1e-10)  # Add small epsilon for numerical stability
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for fractal analysis."""
        try:
            if len(returns) < 10:
                return 0.5
            
            returns = returns.dropna()
            if len(returns) < 10:
                return 0.5
            
            lags = range(2, min(len(returns) // 2, 20))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            
            if len(tau) < 2:
                return 0.5
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def _add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        if 'returns' not in data.columns:
            return data
        
        # Volatility regimes
        vol_20 = data['returns'].rolling(20).std()
        vol_median = vol_20.rolling(252).median()
        data['high_vol_regime'] = (vol_20 > vol_median * 1.5).astype(int)
        
        # Trend regimes
        sma_fast = data['Close'].rolling(20).mean() if 'Close' in data.columns else 0
        sma_slow = data['Close'].rolling(50).mean() if 'Close' in data.columns else 0
        data['bull_regime'] = (sma_fast > sma_slow).astype(int)
        
        return data
    
    def _add_options_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add simulated options-based features."""
        if 'returns' not in data.columns:
            return data
        
        # Simulated implied volatility
        realized_vol = data['returns'].rolling(20).std()
        data['implied_vol_spread'] = realized_vol * np.random.uniform(0.8, 1.2, len(data))
        
        # Simulated put-call ratio
        data['put_call_ratio'] = np.random.uniform(0.5, 1.5, len(data))
        
        return data
    
    async def train_ensemble_models(self, data: pd.DataFrame, target_column: str = 'future_return') -> Dict[str, Any]:
        """Train ensemble of AI models asynchronously."""
        logger.info("Training ensemble models...")
        
        # Prepare features
        features = self.prepare_features(data)
        
        if features.empty or len(features) < 100:
            logger.warning("Insufficient data for training")
            return {}
        
        # Create target variable (future returns)
        if 'Close' in features.columns:
            features['future_return'] = features['Close'].shift(-1) / features['Close'] - 1
        
        # Remove rows with NaN target
        features = features.dropna()
        
        if len(features) < 50:
            logger.warning("Insufficient data after cleaning")
            return {}
        
        X = features[self.feature_columns].values
        y = features[target_column].values
        
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - X_mean) / X_std
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Train models
        training_results = {}
        
        # Train LSTM
        try:
            lstm_result = await self._train_lstm(X_tensor, y_tensor)
            training_results['lstm'] = lstm_result
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
        
        # Train Transformer
        try:
            transformer_result = await self._train_transformer(X_tensor, y_tensor)
            training_results['transformer'] = transformer_result
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
        
        # Train RL Agent (if enough data)
        if len(features) > 200:
            try:
                rl_result = await self._train_rl_agent(features)
                training_results['rl_agent'] = rl_result
            except Exception as e:
                logger.error(f"RL Agent training failed: {e}")
        
        # Store normalization parameters
        self.model_metadata['normalization'] = {
            'mean': X_mean.tolist(),
            'std': X_std.tolist(),
            'feature_columns': self.feature_columns
        }
        
        return training_results
    
    async def _train_lstm(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Train LSTM model."""
        model = self.models['lstm']
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Reshape for LSTM (batch_size, seq_len, input_size)
        X_reshaped = X.unsqueeze(1)  # Add sequence dimension
        
        model.train()
        losses = []
        
        for epoch in range(100):
            optimizer.zero_grad()
            
            predictions, confidence, attention = model(X_reshaped)
            loss = criterion(predictions.squeeze(), y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.6f}")
        
        model.eval()
        return {'final_loss': losses[-1], 'training_losses': losses}
    
    async def _train_transformer(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Train Transformer model."""
        model = self.models['transformer']
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Reshape for Transformer
        X_reshaped = X.unsqueeze(1)
        
        model.train()
        losses = []
        
        for epoch in range(100):
            optimizer.zero_grad()
            
            price_pred, vol_pred, direction_pred = model(X_reshaped)
            loss = criterion(price_pred.squeeze(), y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Transformer Epoch {epoch}, Loss: {loss.item():.6f}")
        
        model.eval()
        return {'final_loss': losses[-1], 'training_losses': losses}
    
    async def _train_rl_agent(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train RL agent for trading decisions."""
        model = self.models['rl_agent']
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Create trading environment simulation
        states = data[self.feature_columns].values
        returns = data['returns'].values if 'returns' in data.columns else np.random.randn(len(data)) * 0.01
        
        # Normalize states
        states_mean = np.mean(states, axis=0)
        states_std = np.std(states, axis=0) + 1e-8
        states_normalized = (states - states_mean) / states_std
        
        model.train()
        total_rewards = []
        
        for episode in range(100):
            episode_reward = 0
            
            for t in range(len(states_normalized) - 1):
                state = torch.FloatTensor(states_normalized[t]).unsqueeze(0).to(self.device)
                
                # Get action probabilities
                q_values = model(state)
                action = torch.argmax(q_values, dim=1).item()
                
                # Calculate reward based on next period return
                next_return = returns[t + 1]
                if action == 0:  # Buy
                    reward = next_return
                elif action == 1:  # Sell
                    reward = -next_return
                else:  # Hold
                    reward = 0
                
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            if episode % 20 == 0:
                logger.info(f"RL Episode {episode}, Reward: {episode_reward:.4f}")
        
        model.eval()
        return {'total_rewards': total_rewards, 'final_reward': total_rewards[-1]}
    
    async def generate_trading_signals(self, data: pd.DataFrame) -> List[MarketSignal]:
        """Generate comprehensive trading signals using AI ensemble."""
        features = self.prepare_features(data)
        
        if features.empty:
            return []
        
        signals = []
        
        try:
            # Get latest features
            latest_features = features[self.feature_columns].iloc[-1:].values
            
            # Normalize using stored parameters
            if 'normalization' in self.model_metadata:
                norm_params = self.model_metadata['normalization']
                latest_features = (latest_features - np.array(norm_params['mean'])) / np.array(norm_params['std'])
            
            X_tensor = torch.FloatTensor(latest_features).to(self.device)
            
            # LSTM signal
            if 'lstm' in self.models:
                lstm_signal = await self._get_lstm_signal(X_tensor)
                signals.append(lstm_signal)
            
            # Transformer signal
            if 'transformer' in self.models:
                transformer_signal = await self._get_transformer_signal(X_tensor)
                signals.append(transformer_signal)
            
            # RL Agent signal
            if 'rl_agent' in self.models:
                rl_signal = await self._get_rl_signal(X_tensor)
                signals.append(rl_signal)
            
            # Technical analysis signal
            technical_signal = self._get_technical_signal(features)
            signals.append(technical_signal)
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _get_lstm_signal(self, X: torch.Tensor) -> MarketSignal:
        """Get signal from LSTM model."""
        model = self.models['lstm']
        model.eval()
        
        with torch.no_grad():
            X_reshaped = X.unsqueeze(1)
            predictions, confidence, _ = model(X_reshaped)
            
            pred_value = predictions.item()
            conf_value = confidence.item()
            
            if pred_value > 0.005:  # 0.5% threshold
                signal_type = 'buy'
                strength = min(abs(pred_value) * 10, 1.0)
            elif pred_value < -0.005:
                signal_type = 'sell'
                strength = min(abs(pred_value) * 10, 1.0)
            else:
                signal_type = 'hold'
                strength = 0.5
        
        return MarketSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=conf_value,
            timeframe='short',
            rationale=f'LSTM prediction: {pred_value:.4f}',
            risk_score=1 - conf_value
        )
    
    async def _get_transformer_signal(self, X: torch.Tensor) -> MarketSignal:
        """Get signal from Transformer model."""
        model = self.models['transformer']
        model.eval()
        
        with torch.no_grad():
            X_reshaped = X.unsqueeze(1)
            price_pred, vol_pred, direction_pred = model(X_reshaped)
            
            direction_probs = direction_pred.squeeze().cpu().numpy()
            up_prob, down_prob, sideways_prob = direction_probs
            
            if up_prob > max(down_prob, sideways_prob):
                signal_type = 'buy'
                strength = up_prob
                confidence = up_prob
            elif down_prob > max(up_prob, sideways_prob):
                signal_type = 'sell'
                strength = down_prob
                confidence = down_prob
            else:
                signal_type = 'hold'
                strength = sideways_prob
                confidence = sideways_prob
        
        return MarketSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timeframe='medium',
            rationale=f'Transformer direction: {signal_type} ({strength:.3f})',
            risk_score=vol_pred.item()
        )
    
    async def _get_rl_signal(self, X: torch.Tensor) -> MarketSignal:
        """Get signal from RL agent."""
        model = self.models['rl_agent']
        model.eval()
        
        with torch.no_grad():
            q_values = model(X)
            action_probs = torch.softmax(q_values, dim=-1)
            action = torch.argmax(q_values, dim=-1).item()
            confidence = torch.max(action_probs).item()
            
            signal_map = {0: 'buy', 1: 'sell', 2: 'hold'}
            signal_type = signal_map[action]
            
            strength = confidence
        
        return MarketSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timeframe='adaptive',
            rationale=f'RL agent action: {action} (Q-value: {q_values.max().item():.3f})',
            risk_score=1 - confidence
        )
    
    def _get_technical_signal(self, features: pd.DataFrame) -> MarketSignal:
        """Get signal from technical analysis."""
        if features.empty:
            return MarketSignal('hold', 0.5, 0.5, 'technical', 'No data', 0.5)
        
        latest = features.iloc[-1]
        
        signals = []
        
        # RSI signal
        if 'rsi_14' in features.columns:
            rsi = latest['rsi_14']
            if rsi > 70:
                signals.append(('sell', 0.7))
            elif rsi < 30:
                signals.append(('buy', 0.7))
        
        # Momentum signal
        if 'momentum_20' in features.columns:
            momentum = latest['momentum_20']
            if momentum > 0.05:
                signals.append(('buy', 0.6))
            elif momentum < -0.05:
                signals.append(('sell', 0.6))
        
        # Aggregate signals
        if not signals:
            signal_type, strength = 'hold', 0.5
        else:
            buy_strength = sum(s[1] for s in signals if s[0] == 'buy') / len(signals)
            sell_strength = sum(s[1] for s in signals if s[0] == 'sell') / len(signals)
            
            if buy_strength > sell_strength:
                signal_type, strength = 'buy', buy_strength
            elif sell_strength > buy_strength:
                signal_type, strength = 'sell', sell_strength
            else:
                signal_type, strength = 'hold', 0.5
        
        return MarketSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=0.8,
            timeframe='technical',
            rationale=f'Technical analysis: {len(signals)} indicators',
            risk_score=0.3
        )

# Initialize the supercharged AI engine
ai_engine = SuperchargedAIEngine()

if __name__ == "__main__":
    logger.info("Supercharged AI Engine initialized")