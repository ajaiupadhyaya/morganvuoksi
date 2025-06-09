"""
Script to run the entire quantitative finance system end-to-end.
"""
import asyncio
import os
import sys
from pathlib import Path
import subprocess
import time
from datetime import datetime
import logging
import yaml
import pandas as pd

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_environment():
    """Set up the environment for running the system."""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('demo_outputs', exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

async def run_data_pipeline():
    """Run the data pipeline."""
    logger.info("Starting data pipeline...")
    from src.data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for symbol in symbols:
        try:
            data = await fetcher.fetch_stock_data(symbol)
            data.to_csv(f'data/{symbol}_data.csv')
            logger.info(f"Fetched data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")

async def train_models():
    """Train all models."""
    logger.info("Starting model training...")
    from src.models.lstm import LSTM
    from src.models.xgboost import XGBoost
    from src.models.arima_garch import ARIMAGARCH
    from src.models.transformer import TransformerModel
    from src.models.rl import PPO
    
    # Load data
    data = pd.read_csv('data/AAPL_data.csv', index_col=0, parse_dates=True)
    
    # Prepare features
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data = data.dropna()
    
    # Train models
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
    
    for model_name, model in models.items():
        try:
            logger.info(f"Training {model_name}...")
            model.fit(data)
            model.save(f'models/{model_name}.pth')
            logger.info(f"Trained and saved {model_name}")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")

def run_backtest():
    """Run backtesting."""
    logger.info("Starting backtesting...")
    try:
        subprocess.run(['python', 'run_backtest.py'], check=True)
        logger.info("Backtesting completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in backtesting: {str(e)}")

def start_dashboard():
    """Start the dashboard."""
    logger.info("Starting dashboard...")
    try:
        subprocess.Popen(['streamlit', 'run', 'src/dashboard/app.py'])
        logger.info("Dashboard started")
    except Exception as e:
        logger.error(f"Error starting dashboard: {str(e)}")

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

async def main():
    """Run the entire system."""
    try:
        # Load configuration
        config = load_config()
        
        # Set up environment
        setup_environment()
        
        # Run data pipeline
        await run_data_pipeline()
        
        # Train models
        await train_models()
        
        # Run backtest
        run_backtest()
        
        # Start dashboard
        start_dashboard()
        
        logger.info("System started successfully!")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    asyncio.run(main()) 