#!/usr/bin/env python3
"""
MorganVuoksi Terminal Demo
This script demonstrates the capabilities of the Bloomberg-style quantitative trading terminal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def create_demo_data():
    """Create sample data for demonstration purposes."""
    
    # Generate sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create realistic price data with trends and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
    
    # Add some trends and patterns
    trend = np.linspace(0, 0.3, len(dates))  # Upward trend
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Annual seasonality
    
    prices = prices * (1 + trend + seasonal)
    
    # Create volume data
    volume = np.random.lognormal(10, 0.5, len(dates))
    
    # Create sample portfolio data
    portfolio_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volume,
        'Returns': np.diff(prices, prepend=prices[0]) / prices,
        'Portfolio_Value': 100000 * np.exp(np.cumsum(returns * 0.8)),  # 80% exposure
        'Cash': 100000 * np.exp(np.cumsum(returns * 0.2)),  # 20% cash
    })
    
    # Add technical indicators
    portfolio_data['MA20'] = portfolio_data['Close'].rolling(window=20).mean()
    portfolio_data['MA50'] = portfolio_data['Close'].rolling(window=50).mean()
    portfolio_data['Volatility'] = portfolio_data['Returns'].rolling(window=20).std()
    portfolio_data['RSI'] = calculate_rsi(portfolio_data['Close'])
    
    return portfolio_data

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def demo_market_data():
    """Demonstrate market data visualization."""
    st.header("📈 Market Data Demo")
    
    data = create_demo_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'],
        name='Price', line=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA20'],
        name='MA20', line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA50'],
        name='MA50', line=dict(color='red')
    ))
    fig.update_layout(
        title="Sample Market Data - Price and Moving Averages",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def demo_ml_predictions():
    """Demonstrate ML predictions."""
    st.header("🤖 ML Predictions Demo")
    
    data = create_demo_data()
    
    # Simulate model predictions
    actual_prices = data['Close'].tail(50)
    dates = data['Date'].tail(50)
    
    # Create mock predictions
    np.random.seed(42)
    lstm_pred = actual_prices * (1 + np.random.normal(0, 0.01, len(actual_prices)))
    xgb_pred = actual_prices * (1 + np.random.normal(0, 0.008, len(actual_prices)))
    transformer_pred = actual_prices * (1 + np.random.normal(0, 0.012, len(actual_prices)))
    
    # Plot predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual_prices,
        name='Actual', line=dict(color='white', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lstm_pred,
        name='LSTM Prediction', line=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=xgb_pred,
        name='XGBoost Prediction', line=dict(color='#ff7f0e')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=transformer_pred,
        name='Transformer Prediction', line=dict(color='#2ca02c')
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        lstm_rmse = np.sqrt(np.mean((lstm_pred - actual_prices) ** 2))
        st.metric("LSTM RMSE", f"{lstm_rmse:.4f}")
    with col2:
        xgb_rmse = np.sqrt(np.mean((xgb_pred - actual_prices) ** 2))
        st.metric("XGBoost RMSE", f"{xgb_rmse:.4f}")
    with col3:
        transformer_rmse = np.sqrt(np.mean((transformer_pred - actual_prices) ** 2))
        st.metric("Transformer RMSE", f"{transformer_rmse:.4f}")

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    st.header("📊 Portfolio Optimization Demo")
    
    # Create sample asset data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    np.random.seed(42)
    
    # Generate correlated returns
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    returns_data = {}
    
    for i, symbol in enumerate(symbols):
        # Create correlated returns
        base_returns = np.random.normal(0.0005, 0.02, len(dates))
        if i > 0:
            # Add correlation with previous asset
            correlation = 0.3 + 0.4 * np.random.random()
            returns_data[symbol] = correlation * returns_data[symbols[i-1]] + \
                                  np.sqrt(1 - correlation**2) * base_returns
        else:
            returns_data[symbol] = base_returns
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Calculate expected returns and covariance
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    # Simulate portfolio weights
    weights = np.random.dirichlet(np.ones(len(symbols)))
    weights_dict = dict(zip(symbols, weights))
    
    # Display portfolio allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimized Portfolio Weights")
        weights_df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': [f"{w:.2%}" for w in weights]
        })
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
        st.subheader("Portfolio Metrics")
        
        # Calculate portfolio metrics
        portfolio_return = sum(weights_dict[s] * expected_returns[s] for s in symbols)
        portfolio_vol = np.sqrt(sum(weights_dict[s] * weights_dict[t] * cov_matrix.loc[s, t] 
                                   for s in symbols for t in symbols))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        st.metric("Expected Return", f"{portfolio_return:.2%}")
        st.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Efficient frontier
        returns_range = np.linspace(0.05, 0.25, 20)
        efficient_frontier = []
        
        for target_ret in returns_range:
            # Simplified efficient frontier calculation
            vol = target_ret * 0.8 + np.random.normal(0, 0.02)
            efficient_frontier.append((vol, target_ret))
        
        frontier_df = pd.DataFrame(efficient_frontier, columns=['Volatility', 'Return'])
        fig = px.scatter(
            frontier_df,
            x='Volatility',
            y='Return',
            title="Efficient Frontier",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

def demo_risk_management():
    """Demonstrate risk management features."""
    st.header("🧪 Risk Management Demo")
    
    # Create sample risk data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Portfolio value with drawdowns
    returns = np.random.normal(0.0005, 0.015, len(dates))
    portfolio_value = 100000 * np.exp(np.cumsum(returns))
    
    # Calculate drawdown
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    
    # VaR calculation
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio VaR (95%)", f"{var_95:.2%}")
    with col2:
        st.metric("Portfolio VaR (99%)", f"{var_99:.2%}")
    with col3:
        st.metric("Current Drawdown", f"{drawdown.iloc[-1]:.2%}")
    with col4:
        st.metric("Max Drawdown", f"{drawdown.min():.2%}")
    
    # Risk charts
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR distribution
        fig = px.histogram(
            x=returns,
            title="Return Distribution & VaR",
            template="plotly_dark"
        )
        fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                     annotation_text="VaR 95%")
        fig.add_vline(x=var_99, line_dash="dash", line_color="orange", 
                     annotation_text="VaR 99%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drawdown chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            yaxis_title="Drawdown (%)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk alerts
    st.subheader("Risk Alerts")
    alerts = [
        {"Level": "Info", "Message": "VaR within normal range", "Time": "2 hours ago"},
        {"Level": "Warning", "Message": "Portfolio correlation approaching threshold", "Time": "1 day ago"},
        {"Level": "Critical", "Message": "Maximum position size exceeded for AAPL", "Time": "3 hours ago"}
    ]
    
    for alert in alerts:
        if alert["Level"] == "Critical":
            st.error(f"🚨 {alert['Message']} ({alert['Time']})")
        elif alert["Level"] == "Warning":
            st.warning(f"⚠️ {alert['Message']} ({alert['Time']})")
        else:
            st.info(f"ℹ️ {alert['Message']} ({alert['Time']})")

def demo_valuation_tools():
    """Demonstrate valuation tools."""
    st.header("📉 Valuation Tools Demo")
    
    st.subheader("Discounted Cash Flow (DCF) Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_fcf = st.number_input("Current Free Cash Flow ($M)", value=1000, key="demo_fcf")
        growth_rate = st.number_input("Growth Rate (%)", value=5.0, step=0.5, key="demo_growth") / 100
        terminal_growth = st.number_input("Terminal Growth (%)", value=2.0, step=0.1, key="demo_terminal") / 100
    
    with col2:
        discount_rate = st.number_input("Discount Rate (%)", value=10.0, step=0.5, key="demo_discount") / 100
        projection_years = st.number_input("Projection Years", value=5, min_value=1, max_value=10, key="demo_years")
        shares_outstanding = st.number_input("Shares Outstanding (M)", value=1000, key="demo_shares")
    
    if st.button("Calculate DCF Value", key="demo_dcf_button"):
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

def main():
    """Main demo function."""
    st.set_page_config(
        page_title="MorganVuoksi Terminal Demo",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stApp {
            background-color: #0e1117;
        }
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 MorganVuoksi Terminal Demo")
    st.markdown("---")
    st.markdown("""
    This demo showcases the key features of the MorganVuoksi Terminal - a Bloomberg-style 
    quantitative trading and research platform. All data shown is simulated for demonstration purposes.
    """)
    
    # Create tabs for different demos
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Market Data", "🤖 ML Predictions", "📊 Portfolio Optimization", 
        "🧪 Risk Management", "📉 Valuation Tools"
    ])
    
    with tab1:
        demo_market_data()
    
    with tab2:
        demo_ml_predictions()
    
    with tab3:
        demo_portfolio_optimization()
    
    with tab4:
        demo_risk_management()
    
    with tab5:
        demo_valuation_tools()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>MorganVuoksi Terminal Demo | Powered by Advanced Quantitative Analytics</p>
            <p>To run the full terminal: <code>streamlit run dashboard/terminal.py</code></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 