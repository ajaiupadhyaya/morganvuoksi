"""
Pytest configuration file.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 1, len(dates)),
        'High': np.random.normal(101, 1, len(dates)),
        'Low': np.random.normal(99, 1, len(dates)),
        'Close': np.random.normal(100, 1, len(dates)),
        'Volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Add technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    return data.dropna()

@pytest.fixture
def model_config():
    """Create sample model configuration."""
    return {
        'lstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'sequence_length': 10
        },
        'xgboost': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'arima_garch': {
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'seasonal': True,
            'm': 12,
            'garch_p': 1,
            'garch_q': 1
        },
        'transformer': {
            'd_model': 64,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'input_dim': 5,
            'output_dim': 1,
            'max_seq_len': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5
        },
        'rl': {
            'state_dim': 5,
            'action_dim': 1,
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'target_kl': 0.01,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5
        }
    }

@pytest.fixture
def backtest_config():
    """Create sample backtest configuration."""
    return {
        'initial_capital': 1000000,
        'transaction_cost': 0.001,
        'slippage': 0.0002,
        'risk_free_rate': 0.02,
        'position_size': 1.0,
        'stop_loss': 0.02,
        'take_profit': 0.04
    }

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)) 