#!/usr/bin/env python3
"""
MorganVuoksi Terminal Demo
Demonstrates key features with simulated data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def generate_demo_data():
    """Generate realistic demo data for the terminal."""
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    demo_data = {}
    
    for symbol in symbols:
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        
        # Start price
        start_price = np.random.uniform(50, 500)
        
        # Generate daily returns with some trend and volatility
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # Slight upward trend
        
        # Add some market events
        event_dates = [50, 150, 250, 350]
        for event_date in event_dates:
            if event_date < len(daily_returns):
                daily_returns[event_date] += np.random.normal(0, 0.05)
        
        # Calculate prices
        prices = [start_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLC data
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Ensure High >= Low
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Add technical indicators
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data['Close'])
        data['Volatility'] = data['Close'].pct_change().rolling(20).std()
        
        demo_data[symbol] = data.set_index('Date')
    
    return demo_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def create_demo_chart(data, symbol):
    """Create demo candlestick chart."""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Moving averages
    if 'MA20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            name='MA20',
            line=dict(color='#ffaa00', width=2)
        ))
    
    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            name='MA50',
            line=dict(color='#0088ff', width=2)
        ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart (Demo Data)",
        yaxis_title='Price',
        template='plotly_dark',
        height=400
    )
    
    return fig

def main():
    """Run the demo."""
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
        .demo-header {
            text-align: center;
            color: #0066cc;
            font-size: 2.5em;
            margin-bottom: 1em;
        }
        .demo-section {
            background-color: #2d2d2d;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="demo-header">🚀 MorganVuoksi Terminal Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Experience the power of institutional-grade quantitative trading")
    
    # Generate demo data
    demo_data = generate_demo_data()
    
    # Demo sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Market Data Demo")
        
        symbol = st.selectbox("Select Symbol", list(demo_data.keys()))
        data = demo_data[symbol]
        
        # Market metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        st.metric("Current Price", f"${current_price:.2f}", 
                 f"{change:+.2f} ({change_pct:+.2f}%)")
        
        # Chart
        fig = create_demo_chart(data, symbol)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Predictions Demo")
        
        # Simulated predictions
        prediction_horizon = st.selectbox("Prediction Horizon", ["1d", "5d", "10d", "30d"])
        
        if st.button("Generate AI Predictions"):
            # Simulate AI predictions
            current_price = data['Close'].iloc[-1]
            volatility = data['Volatility'].iloc[-1]
            
            # Generate realistic prediction
            days = int(prediction_horizon[:-1])
            trend = np.random.normal(0.001, 0.005)  # Slight upward trend
            noise = np.random.normal(0, volatility * np.sqrt(days))
            
            predicted_price = current_price * (1 + trend * days + noise)
            confidence = np.random.uniform(0.6, 0.9)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Price", f"${predicted_price:.2f}")
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col_b:
                signal = "BUY" if predicted_price > current_price else "SELL"
                signal_color = "green" if signal == "BUY" else "red"
                st.markdown(f"<h2 style='color: {signal_color};'>{signal}</h2>", 
                           unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Portfolio optimization demo
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 📈 Portfolio Optimization Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio allocation
        symbols = list(demo_data.keys())
        weights = np.random.dirichlet(np.ones(len(symbols)))
        
        fig = px.pie(
            values=weights,
            names=symbols,
            title="Optimal Portfolio Allocation"
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Portfolio metrics
        st.markdown("#### Portfolio Metrics")
        
        # Calculate portfolio returns
        portfolio_returns = pd.DataFrame()
        for symbol in symbols:
            returns = demo_data[symbol]['Close'].pct_change()
            portfolio_returns[symbol] = returns
        
        portfolio_returns = portfolio_returns.dropna()
        weighted_returns = (portfolio_returns * weights).sum(axis=1)
        
        annual_return = weighted_returns.mean() * 252
        annual_vol = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        st.metric("Expected Return", f"{annual_return:.2%}")
        st.metric("Volatility", f"{annual_vol:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk analysis demo
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Risk Analysis Demo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", "-2.3%")
    with col2:
        st.metric("CVaR (95%)", "-3.1%")
    with col3:
        st.metric("Max Drawdown", "-8.5%")
    with col4:
        st.metric("Beta", "1.2")
    
    # Risk chart
    returns = demo_data['AAPL']['Close'].pct_change().dropna()
    fig = px.histogram(returns, nbins=50, title="Returns Distribution")
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # News sentiment demo
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 📰 News & Sentiment Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Recent News")
        news_items = [
            "Apple Reports Strong Q4 Earnings, Stock Rises 5%",
            "Tech Sector Faces Regulatory Challenges",
            "Federal Reserve Signals Potential Rate Changes",
            "Market Volatility Increases Amid Economic Uncertainty"
        ]
        
        for i, news in enumerate(news_items):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'])
            color = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}[sentiment]
            
            st.markdown(f"<p style='color: {color};'>📰 {news}</p>", 
                       unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Sentiment Analysis")
        
        # Sentiment distribution
        sentiment_data = {
            'Positive': 45,
            'Negative': 25,
            'Neutral': 30
        }
        
        fig = px.pie(
            values=list(sentiment_data.values()),
            names=list(sentiment_data.keys()),
            title="News Sentiment Distribution"
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #1e1e1e; border-radius: 8px;'>
        <h2>🚀 Ready to Experience the Full Terminal?</h2>
        <p>This demo shows just a glimpse of MorganVuoksi Terminal's capabilities.</p>
        <p>Launch the full terminal to access:</p>
        <ul style='text-align: left; display: inline-block;'>
            <li>Real-time market data from multiple sources</li>
            <li>Advanced AI/ML prediction models</li>
            <li>Comprehensive portfolio optimization</li>
            <li>Professional risk management tools</li>
            <li>Interactive backtesting engine</li>
            <li>Reinforcement learning trading agents</li>
            <li>AI-powered news analysis</li>
            <li>Automated reporting and AI assistant</li>
        </ul>
        <br>
        <p><strong>Run: <code>./run_terminal.sh</code> to launch the full MorganVuoksi Terminal</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 