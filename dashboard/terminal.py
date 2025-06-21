<<<<<<< HEAD
"""
MorganVuoksi - Bloomberg-Style Quantitative Trading Terminal
A comprehensive Streamlit dashboard for quantitative research and trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import all our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.pipeline import DataPipeline
from src.models.lstm import LSTM
from src.models.xgboost import XGBoost
from src.models.transformer import Transformer
from src.models.arima_garch import ARIMAGARCH
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.risk import RiskManager
from src.signals.signal_generator import SignalGenerator
from src.signals.alpha_factors import AlphaFactors
from src.signals.ml_models import MLSignalGenerator
from src.backtesting.engine import BacktestEngine
from src.execution.simulate import ExecutionSimulator
from src.ml.ecosystem import MLEcosystem
from src.ml.learning_loop import LearningLoop
from src.visuals import (
    plot_equity_curve, plot_rolling_metrics, plot_trade_annotations,
    plot_risk_heatmap, plot_strategy_comparison, plot_signal_strength,
    plot_feature_importance, plot_signal_decay, plot_risk_decomposition,
    plot_correlation_matrix, plot_drawdown_analysis
)

# Page configuration
st.set_page_config(
    page_title="MorganVuoksi Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg-style dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    .metric-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #4a4a4a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        color: #fafafa;
        border: 1px solid #4a4a4a;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'models_cache' not in st.session_state:
    st.session_state.models_cache = {}
if 'portfolio_cache' not in st.session_state:
    st.session_state.portfolio_cache = {}

# Load configuration
@st.cache_resource
def load_config():
    return Config()

config = load_config()

# Data fetching and caching
@st.cache_data(ttl=300)
def fetch_market_data(symbol: str, period: str = "1y"):
    """Fetch market data with caching."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Sidebar configuration
st.sidebar.title("🎯 MorganVuoksi Terminal")
st.sidebar.markdown("---")

# Symbol and date selection
symbol = st.sidebar.text_input("Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# Strategy selection
st.sidebar.markdown("### 📊 Strategy Configuration")
strategy = st.sidebar.selectbox(
    "Trading Strategy",
    ["Momentum", "Mean Reversion", "ML Ensemble", "Risk Parity", "Custom"]
)

# Model selection
st.sidebar.markdown("### 🤖 ML Models")
selected_models = st.sidebar.multiselect(
    "Select Models",
    ["LSTM", "XGBoost", "Transformer", "ARIMA-GARCH"],
    default=["LSTM", "XGBoost"]
)

# Risk parameters
st.sidebar.markdown("### ⚠️ Risk Management")
max_position_size = st.sidebar.slider("Max Position Size (%)", 1, 20, 5)
stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 10, 3)
max_drawdown = st.sidebar.slider("Max Drawdown (%)", 5, 30, 15)

# Main dashboard
st.title("📈 MorganVuoksi Quantitative Trading Terminal")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📈 Market Data", "🤖 AI/ML Predictions", "⚙️ Backtesting", "📊 Portfolio Optimizer",
    "🧠 NLP & Sentiment", "📉 Valuation Tools", "💸 Trade Simulator", "🧾 Reports",
    "🧪 Risk Management", "🧬 LLM Assistant"
])

# Tab 1: Market Data Viewer
with tab1:
    st.header("📈 Market Data Viewer")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch data
    data = fetch_market_data(symbol, period)
    
    if data is not None:
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            daily_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
            st.metric("Daily Return", f"{daily_return:.2f}%")
        with col3:
            volatility = data['Volatility'].iloc[-1] * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        with col4:
            rsi = data['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi:.1f}")
        
        # Price chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Recent Data")
        st.dataframe(data.tail(20), use_container_width=True)

# Tab 2: AI/ML Predictions
with tab2:
    st.header("🤖 AI/ML Predictions")
    
    if st.button("🔄 Train Models & Generate Predictions"):
        with st.spinner("Training models..."):
            if data is not None:
                # Prepare features
                feature_cols = ['Returns', 'Volatility', 'MA20', 'MA50', 'RSI']
                X = data[feature_cols].dropna()
                y = data['Close'].shift(-1).dropna()
                
                # Align data
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                predictions = {}
                model_metrics = {}
                
                # Train selected models
                if "LSTM" in selected_models:
                    try:
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
                        predictions['LSTM'] = lstm_pred
                        model_metrics['LSTM'] = np.sqrt(np.mean((lstm_pred - y_test) ** 2))
                    except Exception as e:
                        st.error(f"LSTM training failed: {str(e)}")
                
                if "XGBoost" in selected_models:
                    try:
                        xgb = XGBoost(config={
                            'max_depth': 6,
                            'learning_rate': 0.1,
                            'n_estimators': 100,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8
                        })
                        xgb.fit(X_train, y_train)
                        xgb_pred = xgb.predict(X_test)
                        predictions['XGBoost'] = xgb_pred
                        model_metrics['XGBoost'] = np.sqrt(np.mean((xgb_pred - y_test) ** 2))
                    except Exception as e:
                        st.error(f"XGBoost training failed: {str(e)}")
                
                if "Transformer" in selected_models:
                    try:
                        transformer = Transformer(config={
                            'd_model': 64,
                            'n_heads': 4,
                            'n_layers': 2,
                            'dropout': 0.1,
                            'batch_size': 32,
                            'epochs': 50,
                            'learning_rate': 0.001
                        })
                        transformer.fit(X_train, y_train)
                        transformer_pred = transformer.predict(X_test)
                        predictions['Transformer'] = transformer_pred
                        model_metrics['Transformer'] = np.sqrt(np.mean((transformer_pred - y_test) ** 2))
                    except Exception as e:
                        st.error(f"Transformer training failed: {str(e)}")
                
                if "ARIMA-GARCH" in selected_models:
                    try:
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
                        predictions['ARIMA-GARCH'] = arima_pred
                        model_metrics['ARIMA-GARCH'] = np.sqrt(np.mean((arima_pred - y_test) ** 2))
                    except Exception as e:
                        st.error(f"ARIMA-GARCH training failed: {str(e)}")
                
                # Display predictions
                if predictions:
                    # Prediction chart
                    fig = go.Figure()
                    
                    # Actual values
                    fig.add_trace(go.Scatter(
                        x=X_test.index, y=y_test,
                        name='Actual', line=dict(color='white', width=2)
                    ))
                    
                    # Model predictions
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    for i, (model_name, pred) in enumerate(predictions.items()):
                        fig.add_trace(go.Scatter(
                            x=X_test.index, y=pred,
                            name=f'{model_name} Prediction',
                            line=dict(color=colors[i % len(colors)])
                        ))
                    
                    fig.update_layout(
                        title="Model Predictions vs Actual",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_dark",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model performance metrics
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for i, (model_name, rmse) in enumerate(model_metrics.items()):
                        with [col1, col2, col3, col4][i % 4]:
                            st.metric(f"{model_name} RMSE", f"{rmse:.4f}")
                    
                    # Feature importance (if XGBoost is available)
                    if "XGBoost" in predictions:
                        try:
                            importance_df = xgb.get_feature_importance()
                            fig_importance = px.bar(
                                importance_df,
                                x='feature',
                                y='importance',
                                title="Feature Importance (XGBoost)",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                        except:
                            pass

# Tab 3: Backtesting Engine
with tab3:
    st.header("⚙️ Backtesting Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_start = st.date_input("Start Date", value=data.index[0].date() if data is not None else datetime.now().date())
        backtest_end = st.date_input("End Date", value=data.index[-1].date() if data is not None else datetime.now().date())
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    
    with col2:
        commission = st.number_input("Commission (%)", value=0.1, step=0.01)
        slippage = st.number_input("Slippage (%)", value=0.05, step=0.01)
        rebalance_freq = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly"])
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Running backtest..."):
            if data is not None:
                try:
                    # Create backtest engine
                    backtest_engine = BacktestEngine(
                        initial_capital=initial_capital,
                        commission=commission/100,
                        slippage=slippage/100
                    )
                    
                    # Simple momentum strategy
                    data_copy = data.copy()
                    data_copy['Signal'] = 0
                    data_copy.loc[data_copy['Returns'] > 0, 'Signal'] = 1
                    data_copy.loc[data_copy['Returns'] < 0, 'Signal'] = -1
                    
                    # Run backtest
                    results = backtest_engine.run_backtest(
                        data=data_copy,
                        strategy_name="Momentum Strategy"
                    )
                    
                    # Display results
                    st.subheader("Backtest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{results['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    with col4:
                        st.metric("Final Portfolio", f"${results['final_portfolio']:,.0f}")
                    
                    # Equity curve
                    if 'equity_curve' in results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['equity_curve'].index,
                            y=results['equity_curve']['Portfolio Value'],
                            name='Portfolio Value',
                            line=dict(color='#1f77b4')
                        ))
                        fig.update_layout(
                            title="Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade analysis
                    if 'trades' in results and len(results['trades']) > 0:
                        st.subheader("Trade Analysis")
                        trades_df = pd.DataFrame(results['trades'])
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Trade distribution
                        fig = px.histogram(
                            trades_df,
                            x='return',
                            title="Trade Return Distribution",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")

# Tab 4: Portfolio Optimizer
with tab4:
    st.header("📊 Portfolio Optimizer")
    
    # Portfolio configuration
    col1, col2 = st.columns(2)
    
    with col1:
        symbols = st.text_area("Symbols (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,TSLA")
        symbols_list = [s.strip().upper() for s in symbols.split(",")]
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Mean-Variance", "Risk Parity", "Maximum Sharpe", "Minimum Variance"]
        )
    
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
        target_return = st.number_input("Target Return (%)", value=10.0, step=0.5) / 100
        max_weight = st.number_input("Max Weight per Asset (%)", value=30.0, step=5.0) / 100
    
    if st.button("🎯 Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            try:
                # Fetch data for all symbols
                portfolio_data = {}
                for symbol in symbols_list:
                    data = fetch_market_data(symbol, "1y")
                    if data is not None:
                        portfolio_data[symbol] = data['Returns'].dropna()
                
                if len(portfolio_data) > 1:
                    # Create returns dataframe
                    returns_df = pd.DataFrame(portfolio_data)
                    returns_df = returns_df.dropna()
                    
                    # Calculate expected returns and covariance
                    expected_returns = returns_df.mean() * 252  # Annualized
                    cov_matrix = returns_df.cov() * 252  # Annualized
                    
                    # Create portfolio optimizer
                    optimizer = PortfolioOptimizer(
                        expected_returns=expected_returns,
                        cov_matrix=cov_matrix,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Optimize based on method
                    if optimization_method == "Mean-Variance":
                        weights = optimizer.optimize_mean_variance(target_return=target_return)
                    elif optimization_method == "Risk Parity":
                        weights = optimizer.optimize_risk_parity()
                    elif optimization_method == "Maximum Sharpe":
                        weights = optimizer.optimize_max_sharpe()
                    elif optimization_method == "Minimum Variance":
                        weights = optimizer.optimize_min_variance()
                    
                    # Display results
                    st.subheader("Optimized Portfolio")
                    
                    # Portfolio weights
                    weights_df = pd.DataFrame({
                        'Symbol': list(weights.keys()),
                        'Weight': [f"{w:.2%}" for w in weights.values()]
                    })
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Pie chart
                        fig = px.pie(
                            weights_df,
                            values=[float(w.strip('%')) for w in weights_df['Weight']],
                            names=weights_df['Symbol'],
                            title="Portfolio Allocation",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Portfolio metrics
                        portfolio_return = sum(weights[s] * expected_returns[s] for s in weights.keys())
                        portfolio_vol = np.sqrt(sum(weights[s] * weights[t] * cov_matrix.loc[s, t] 
                                                   for s in weights.keys() for t in weights.keys()))
                        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                        
                        st.metric("Expected Return", f"{portfolio_return:.2%}")
                        st.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                        # Efficient frontier
                        returns_range = np.linspace(0.05, 0.25, 50)
                        efficient_frontier = []
                        
                        for target_ret in returns_range:
                            try:
                                w = optimizer.optimize_mean_variance(target_return=target_ret)
                                vol = np.sqrt(sum(w[s] * w[t] * cov_matrix.loc[s, t] 
                                                 for s in w.keys() for t in w.keys()))
                                efficient_frontier.append((vol, target_ret))
                            except:
                                pass
                        
                        if efficient_frontier:
                            frontier_df = pd.DataFrame(efficient_frontier, columns=['Volatility', 'Return'])
                            fig = px.scatter(
                                frontier_df,
                                x='Volatility',
                                y='Return',
                                title="Efficient Frontier",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("Need at least 2 symbols with valid data for portfolio optimization")
            
            except Exception as e:
                st.error(f"Portfolio optimization failed: {str(e)}")

# Tab 5: NLP & Sentiment
with tab5:
    st.header("🧠 NLP & Sentiment Analysis")
    
    # Mock sentiment data (in production, this would come from real NLP models)
    sentiment_data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'Sentiment_Score': np.random.normal(0, 0.5, 100),
        'News_Volume': np.random.poisson(50, 100),
        'Social_Media_Score': np.random.normal(0, 0.3, 100)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sentiment_data['Date'],
            y=sentiment_data['Sentiment_Score'],
            name='Sentiment Score',
            line=dict(color='#1f77b4')
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Sentiment Score Over Time",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("News Volume vs Sentiment")
        fig = px.scatter(
            sentiment_data,
            x='News_Volume',
            y='Sentiment_Score',
            title="News Volume vs Sentiment",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment metrics
    st.subheader("Sentiment Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Sentiment", f"{sentiment_data['Sentiment_Score'].iloc[-1]:.3f}")
    with col2:
        st.metric("Sentiment Trend", f"{sentiment_data['Sentiment_Score'].diff().mean():.3f}")
    with col3:
        st.metric("News Volume", f"{sentiment_data['News_Volume'].iloc[-1]:.0f}")
    with col4:
        st.metric("Social Score", f"{sentiment_data['Social_Media_Score'].iloc[-1]:.3f}")

# Tab 6: Valuation Tools
with tab6:
    st.header("📉 Valuation Tools")
    
    valuation_method = st.selectbox(
        "Valuation Method",
        ["DCF Model", "Comparable Analysis", "LBO Model", "Dividend Discount Model"]
    )
    
    if valuation_method == "DCF Model":
        st.subheader("Discounted Cash Flow Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_fcf = st.number_input("Current Free Cash Flow ($M)", value=1000)
            growth_rate = st.number_input("Growth Rate (%)", value=5.0, step=0.5) / 100
            terminal_growth = st.number_input("Terminal Growth (%)", value=2.0, step=0.1) / 100
        
        with col2:
            discount_rate = st.number_input("Discount Rate (%)", value=10.0, step=0.5) / 100
            projection_years = st.number_input("Projection Years", value=5, min_value=1, max_value=10)
            shares_outstanding = st.number_input("Shares Outstanding (M)", value=1000)
        
        if st.button("Calculate DCF Value"):
            # DCF calculation
            fcf_forecast = []
            for year in range(1, projection_years + 1):
                fcf = current_fcf * (1 + growth_rate) ** year
                fcf_forecast.append(fcf)
            
            # Terminal value
            terminal_fcf = fcf_forecast[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # Present values
            pv_fcf = sum(fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(fcf_forecast))
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
            
            enterprise_value = pv_fcf + pv_terminal
            equity_value = enterprise_value / shares_outstanding
            
            st.subheader("DCF Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Enterprise Value", f"${enterprise_value:,.0f}M")
            with col2:
                st.metric("Equity Value", f"${equity_value:,.2f}")
            with col3:
                st.metric("Per Share Value", f"${equity_value:,.2f}")
            
            # FCF forecast chart
            forecast_df = pd.DataFrame({
                'Year': range(1, projection_years + 1),
                'FCF': fcf_forecast
            })
            
            fig = px.bar(
                forecast_df,
                x='Year',
                y='FCF',
                title="Free Cash Flow Forecast",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 7: Trade Simulator
with tab7:
    st.header("💸 Trade Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trade_symbol = st.text_input("Trade Symbol", value=symbol)
        trade_side = st.selectbox("Trade Side", ["Buy", "Sell"])
        trade_size = st.number_input("Trade Size (shares)", value=100, min_value=1)
        trade_price = st.number_input("Trade Price ($)", value=data['Close'].iloc[-1] if data is not None else 100.0)
    
    with col2:
        order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop", "Stop Limit"])
        time_in_force = st.selectbox("Time in Force", ["Day", "GTC", "IOC"])
        algo_type = st.selectbox("Algorithm", ["TWAP", "VWAP", "POV", "Custom"])
    
    if st.button("📊 Simulate Trade"):
        with st.spinner("Simulating trade execution..."):
            try:
                # Create execution simulator
                simulator = ExecutionSimulator(
                    market_data=data if data is not None else pd.DataFrame(),
                    commission_rate=0.001,
                    slippage_model="linear"
                )
                
                # Simulate trade
                execution_result = simulator.simulate_trade(
                    symbol=trade_symbol,
                    side=trade_side.lower(),
                    quantity=trade_size,
                    price=trade_price,
                    order_type=order_type.lower(),
                    algo_type=algo_type.lower()
                )
                
                st.subheader("Trade Execution Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Execution Price", f"${execution_result['avg_price']:.2f}")
                with col2:
                    st.metric("Total Cost", f"${execution_result['total_cost']:,.2f}")
                with col3:
                    st.metric("Market Impact", f"${execution_result['market_impact']:.2f}")
                with col4:
                    st.metric("Execution Time", f"{execution_result['execution_time']:.2f}s")
                
                # Execution timeline
                if 'execution_timeline' in execution_result:
                    timeline_df = pd.DataFrame(execution_result['execution_timeline'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=timeline_df['timestamp'],
                        y=timeline_df['price'],
                        mode='lines+markers',
                        name='Execution Price',
                        line=dict(color='#1f77b4')
                    ))
                    fig.update_layout(
                        title="Trade Execution Timeline",
                        xaxis_title="Time",
                        yaxis_title="Price ($)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Trade simulation failed: {str(e)}")

# Tab 8: Report Generator
with tab8:
    st.header("🧾 Report Generator")
    
    report_type = st.selectbox(
        "Report Type",
        ["Portfolio Performance", "Risk Analysis", "Strategy Backtest", "Market Analysis"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
        report_end_date = st.date_input("End Date", value=datetime.now().date())
    
    with col2:
        include_charts = st.checkbox("Include Charts", value=True)
        include_tables = st.checkbox("Include Tables", value=True)
        export_format = st.selectbox("Export Format", ["PDF", "HTML", "Excel"])
    
    if st.button("📄 Generate Report"):
        with st.spinner("Generating report..."):
            st.success("Report generated successfully!")
            
            # Mock report content
            st.subheader("Executive Summary")
            st.write("""
            This report provides a comprehensive analysis of the portfolio performance and market conditions
            for the specified period. Key highlights include positive returns across all strategies with
            strong risk-adjusted performance metrics.
            """)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", "12.5%")
            with col2:
                st.metric("Sharpe Ratio", "1.85")
            with col3:
                st.metric("Max Drawdown", "-3.2%")
            with col4:
                st.metric("Volatility", "8.7%")
            
            # Risk analysis
            st.subheader("Risk Analysis")
            risk_data = pd.DataFrame({
                'Metric': ['VaR (95%)', 'CVaR (95%)', 'Beta', 'Alpha'],
                'Value': ['-2.1%', '-3.5%', '0.95', '2.3%']
            })
            st.dataframe(risk_data, use_container_width=True)

# Tab 9: Risk Management Dashboard
with tab9:
    st.header("🧪 Risk Management Dashboard")
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio VaR (95%)", "-2.1%")
    with col2:
        st.metric("Current Drawdown", "-1.5%")
    with col3:
        st.metric("Portfolio Beta", "0.95")
    with col4:
        st.metric("Correlation", "0.65")
    
    # Risk charts
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR distribution
        var_data = np.random.normal(-0.02, 0.01, 1000)
        fig = px.histogram(
            x=var_data,
            title="Value at Risk Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drawdown chart
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        drawdown = np.cumsum(np.random.normal(0, 0.01, 100))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk alerts
    st.subheader("Risk Alerts")
    
    alerts = [
        {"Level": "Warning", "Message": "Portfolio correlation approaching threshold", "Time": "2 hours ago"},
        {"Level": "Info", "Message": "VaR within normal range", "Time": "1 day ago"},
        {"Level": "Critical", "Message": "Maximum position size exceeded for AAPL", "Time": "3 hours ago"}
    ]
    
    for alert in alerts:
        if alert["Level"] == "Critical":
            st.error(f"🚨 {alert['Message']} ({alert['Time']})")
        elif alert["Level"] == "Warning":
            st.warning(f"⚠️ {alert['Message']} ({alert['Time']})")
        else:
            st.info(f"ℹ️ {alert['Message']} ({alert['Time']})")

# Tab 10: LLM Assistant
with tab10:
    st.header("🧬 LLM Assistant")
    
    # Chat interface
    st.subheader("Chat with MorganVuoksi AI")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about markets, strategies, or analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            response = generate_ai_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_ai_response(prompt: str) -> str:
    """Generate AI response based on user prompt."""
    # Mock AI responses (in production, this would use a real LLM API)
    responses = {
        "market": "Based on current market conditions, I recommend a defensive positioning with focus on quality stocks and increased cash allocation.",
        "strategy": "For the current market regime, a momentum strategy with risk management overlay would be optimal. Consider 60% equity, 30% bonds, 10% alternatives.",
        "risk": "Current portfolio risk metrics indicate moderate exposure. VaR is within acceptable limits, but monitor correlation levels closely.",
        "analysis": "Technical indicators suggest a bullish bias with support at key levels. Fundamental analysis shows strong earnings growth potential."
    }
    
    prompt_lower = prompt.lower()
    
    if "market" in prompt_lower:
        return responses["market"]
    elif "strategy" in prompt_lower:
        return responses["strategy"]
    elif "risk" in prompt_lower:
        return responses["risk"]
    elif "analysis" in prompt_lower:
        return responses["analysis"]
    else:
        return "I'm here to help with your quantitative analysis needs. You can ask me about market conditions, trading strategies, risk management, or portfolio analysis."

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>MorganVuoksi Terminal v1.0 | Powered by Advanced Quantitative Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
) 
