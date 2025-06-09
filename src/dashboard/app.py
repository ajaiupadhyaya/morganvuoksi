"""
Dashboard application for visualizing model results.
"""
import asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from ..data.fetcher import DataFetcher
from ..models.lstm import LSTM
from ..models.xgboost import XGBoost
from ..models.arima_garch import ARIMAGARCH
import numpy as np

async def fetch_and_prepare_data(symbol: str, days: int = 365):
    """Fetch and prepare data for modeling."""
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

def train_models(data: pd.DataFrame):
    """Train all models on the data."""
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
    
    # Train LSTM
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
    
    # Train XGBoost
    xgb_model = XGBoost(config={
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Train ARIMA-GARCH
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
    
    return {
        'test_data': test_data,
        'y_test': y_test,
        'lstm_pred': lstm_pred,
        'xgb_pred': xgb_pred,
        'arima_pred': arima_pred,
        'vol_pred': vol_pred,
        'xgb_importance': xgb_model.get_feature_importance()
    }

def create_plots(results: dict):
    """Create interactive plots for the dashboard."""
    # Price and predictions plot
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('Price', 'Volatility'))
    
    # Add price
    fig1.add_trace(
        go.Scatter(x=results['test_data'].index, y=results['test_data']['Close'],
                  name='Actual Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add predictions
    fig1.add_trace(
        go.Scatter(x=results['test_data'].index, y=results['lstm_pred'],
                  name='LSTM', line=dict(color='red')),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=results['test_data'].index, y=results['xgb_pred'],
                  name='XGBoost', line=dict(color='green')),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=results['test_data'].index, y=results['arima_pred'],
                  name='ARIMA-GARCH', line=dict(color='purple')),
        row=1, col=1
    )
    
    # Add volatility
    fig1.add_trace(
        go.Scatter(x=results['test_data'].index, y=results['vol_pred'],
                  name='Predicted Volatility', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig1.update_layout(height=800, title_text="Price Predictions and Volatility")
    
    # Feature importance plot
    fig2 = go.Figure(data=[
        go.Bar(x=results['xgb_importance']['feature'],
               y=results['xgb_importance']['importance'])
    ])
    fig2.update_layout(title_text="XGBoost Feature Importance")
    
    return fig1, fig2

def main():
    """Run the dashboard."""
    st.set_page_config(page_title="Quantitative Finance Dashboard", layout="wide")
    
    st.title("Quantitative Finance Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    symbol = st.sidebar.text_input("Symbol", value="AAPL")
    days = st.sidebar.slider("Days of Data", min_value=30, max_value=365, value=365)
    
    if st.sidebar.button("Update"):
        with st.spinner("Fetching data and training models..."):
            # Fetch data
            data = asyncio.run(fetch_and_prepare_data(symbol, days))
            
            # Train models
            results = train_models(data)
            
            # Create plots
            fig1, fig2 = create_plots(results)
            
            # Display plots
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display metrics
            st.header("Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("LSTM RMSE",
                         f"{np.sqrt(np.mean((results['lstm_pred'] - results['y_test']) ** 2)):.4f}")
            
            with col2:
                st.metric("XGBoost RMSE",
                         f"{np.sqrt(np.mean((results['xgb_pred'] - results['y_test']) ** 2)):.4f}")
            
            with col3:
                st.metric("ARIMA-GARCH RMSE",
                         f"{np.sqrt(np.mean((results['arima_pred'] - results['y_test']) ** 2)):.4f}")

if __name__ == '__main__':
    main()
