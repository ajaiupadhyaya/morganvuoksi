"""
Main script to run the quantitative finance system.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from src.data.fetcher import DataFetcher
from src.models.lstm import LSTM
from src.models.xgboost import XGBoost
from src.models.arima_garch import ARIMAGARCH
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

async def run_system():
    """Run the complete quantitative finance system."""
    try:
        # Initialize components
        fetcher = DataFetcher()
        
        # Fetch data
        symbol = 'AAPL'
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        
        logger.info(f"Fetching {symbol} data...")
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
        
        # Drop NaN values
        data = data.dropna()
        
        # Split data
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Prepare features and target
        feature_cols = ['Returns', 'Volatility', 'MA20', 'MA50', 'RSI']
        X_train = train_data[feature_cols]
        y_train = train_data['Close'].shift(-1).dropna()
        X_test = test_data[feature_cols]
        y_test = test_data['Close'].shift(-1).dropna()
        
        # Train models
        logger.info("Training LSTM model...")
        lstm = LSTM(config={
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'sequence_length': 10
        })
        lstm.fit(X_train, y_train)
        lstm_pred = lstm.predict(X_test)
        
        logger.info("Training XGBoost model...")
        xgb_model = XGBoost(config={
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        })
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        logger.info("Training ARIMA-GARCH model...")
        arima_garch = ARIMAGARCH(config={
            'max_p': 5,
            'max_d': 2,
            'max_q': 5,
            'seasonal': True,
            'm': 12,
            'garch_p': 1,
            'garch_q': 1
        })
        arima_garch.fit(X_train, y_train)
        arima_pred, vol_pred = arima_garch.predict(X_test, horizon=len(X_test))
        
        # Evaluate models
        logger.info("\nModel Evaluation:")
        logger.info(f"LSTM RMSE: {np.sqrt(np.mean((lstm_pred - y_test) ** 2)):.4f}")
        logger.info(f"XGBoost RMSE: {np.sqrt(np.mean((xgb_pred - y_test) ** 2)):.4f}")
        logger.info(f"ARIMA-GARCH RMSE: {np.sqrt(np.mean((arima_pred - y_test) ** 2)):.4f}")
        
        # Show feature importance
        logger.info("\nXGBoost Feature Importance:")
        importance = xgb_model.get_feature_importance()
        logger.info(importance)
        
        # Save models
        logger.info("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        lstm.save('models/lstm.pth')
        xgb_model.save('models/xgb.json')
        arima_garch.save('models/arima_garch')
        
        logger.info("System run completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running system: {str(e)}")
        raise

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == '__main__':
    asyncio.run(run_system()) 