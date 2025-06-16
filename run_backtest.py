"""
Script to run backtest with all models.
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.data.fetcher import DataFetcher
from src.models.lstm import LSTM
from src.models.xgboost import XGBoost
from src.models.arima_garch import ARIMAGARCH
from src.models.transformer import TransformerModel
from src.models.rl import PPO
from src.backtesting.engine import BacktestEngine
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

async def prepare_data(symbol: str, days: int = 365):
    """Fetch and prepare data for backtesting."""
    fetcher = DataFetcher()
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    data = await fetcher.fetch_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Prepare features
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    return data.dropna()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals(data: pd.DataFrame, model, model_type: str) -> pd.Series:
    """Generate trading signals from model predictions."""
    feature_cols = ['Returns', 'Volatility', 'MA20', 'MA50', 'RSI']
    X = data[feature_cols].values
    
    if model_type == 'lstm':
        predictions = model.predict(X)
        signals = pd.Series(0, index=data.index)
        signals[predictions > 0.5] = 1
        signals[predictions < -0.5] = -1
    
    elif model_type == 'xgboost':
        predictions = model.predict(X)
        signals = pd.Series(0, index=data.index)
        signals[predictions > 0.5] = 1
        signals[predictions < -0.5] = -1
    
    elif model_type == 'arima_garch':
        predictions, _ = model.predict(X, horizon=len(X))
        signals = pd.Series(0, index=data.index)
        signals[predictions > 0] = 1
        signals[predictions < 0] = -1
    
    elif model_type == 'transformer':
        predictions = model.predict(X)
        signals = pd.Series(0, index=data.index)
        signals[predictions > 0] = 1
        signals[predictions < 0] = -1
    
    elif model_type == 'rl':
        predictions = model.predict(X)
        signals = pd.Series(0, index=data.index)
        signals[predictions > 0.5] = 1
        signals[predictions < -0.5] = -1
    
    return signals

async def run_backtest():
    """Run backtest with all models."""
    # Prepare data
    symbol = 'AAPL'
    data = await prepare_data(symbol, days=365)
    
    # Initialize models
    models = {
        'lstm': LSTM(config={
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'sequence_length': 10
        }),
        'xgboost': XGBoost(config={
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }),
        'arima_garch': ARIMAGARCH(config={
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'seasonal': True,
            'm': 12,
            'garch_p': 1,
            'garch_q': 1
        }),
        'transformer': TransformerModel(config={
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
        }),
        'rl': PPO(config={
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
        })
    }
    
    # Initialize backtest engine
    backtest = BacktestEngine(config={
        'initial_capital': 1000000,
        'transaction_cost': 0.001,
        'slippage': 0.0002,
        'risk_free_rate': 0.02,
        'position_size': 1.0,
        'stop_loss': 0.02,
        'take_profit': 0.04
    })
    
    # Run backtest for each model
    results = {}
    for model_name, model in models.items():
        logger.info(f"Running backtest for {model_name}...")
        
        # Generate signals
        signals = generate_signals(data, model, model_name)
        
        # Run backtest
        result = backtest.run_backtest(data, signals)
        results[model_name] = result
        
        # Generate report
        report = backtest.generate_report(result, save_path=f'demo_outputs/{model_name}_report.txt')
        logger.info(f"\n{report}")
        
        # Plot results
        backtest.plot_results(result, save_path=f'demo_outputs/{model_name}_results.png')
    
    # Compare models
    comparison = pd.DataFrame({
        model_name: result['metrics']
        for model_name, result in results.items()
    }).T
    
    comparison.to_csv('demo_outputs/model_comparison.csv')
    logger.info("\nModel Comparison:")
    logger.info(comparison)

if __name__ == '__main__':
    asyncio.run(run_backtest()) 
