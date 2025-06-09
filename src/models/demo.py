"""
Demo script for the ML models.
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..data.fetcher import DataFetcher
from .lstm import LSTM
from .xgboost import XGBoost
from .arima_garch import ARIMAGARCH

async def main():
    """Run model demo."""
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Fetch data
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nFetching {symbol} data...")
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
    
    # Train and evaluate LSTM
    print("\nTraining LSTM model...")
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
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = XGBoost(config={
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Train and evaluate ARIMA-GARCH
    print("\nTraining ARIMA-GARCH model...")
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
    print("\nModel Evaluation:")
    print(f"LSTM RMSE: {np.sqrt(np.mean((lstm_pred - y_test) ** 2)):.4f}")
    print(f"XGBoost RMSE: {np.sqrt(np.mean((xgb_pred - y_test) ** 2)):.4f}")
    print(f"ARIMA-GARCH RMSE: {np.sqrt(np.mean((arima_pred - y_test) ** 2)):.4f}")
    
    # Show feature importance for XGBoost
    print("\nXGBoost Feature Importance:")
    importance = xgb_model.get_feature_importance()
    print(importance)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == '__main__':
    asyncio.run(main()) 