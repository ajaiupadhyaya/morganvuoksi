"""
MorganVuoksi Terminal - Simplified Version for Railway Deployment
Professional trading interface with minimal dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MorganVuoksi Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Bloomberg Terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
    
    body {
        font-family: 'Roboto Mono', monospace;
        color: #E0E0E0;
        background-color: #000000;
    }

    .main {
        background-color: #000000;
    }

    .stApp {
        background-color: #000000;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
    }
    
    h2 {
         color: #FFA500;
         border-bottom: 1px solid #333;
         padding-bottom: 5px;
         margin-top: 1.5rem;
         margin-bottom: 1rem;
    }

    .stSidebar {
        background-color: #0A0A0A;
        border-right: 1px solid #222;
    }

    .stSidebar h2, .stSidebar h3 {
         color: #FFA500;
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #1A1A1A;
        color: #E0E0E0;
        border: 1px solid #444;
        border-radius: 2px;
    }

    .stButton > button {
        background-color: #FFA500;
        color: #000000;
        font-weight: 700;
        border-radius: 2px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FFC14D;
    }
</style>
""", unsafe_allow_html=True)

class MorganVuoksiTerminal:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        
    def run(self):
        """Main application runner."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Render the terminal header."""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; border-bottom: 2px solid #FFA500;">
            <h1 style="color: #FFA500; font-size: 3rem; margin: 0;">MORGANVUOKSI TERMINAL</h1>
            <p style="color: #888; font-size: 1.2rem; margin: 0.5rem 0;">Professional Quantitative Trading Platform</p>
            <p style="color: #00FF00; font-size: 0.9rem; margin: 0;">● LIVE DATA STREAM</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar controls."""
        with st.sidebar:
            st.header("📊 MARKET CONTROLS")
            
            # Symbol selection
            selected_symbol = st.selectbox(
                "Select Symbol",
                self.symbols,
                index=0
            )
            
            # Time period
            period = st.selectbox(
                "Time Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                index=5
            )
            
            # Update button
            if st.button("🔄 UPDATE DATA", type="primary"):
                st.rerun()
            
            st.markdown("---")
            st.header("⚙️ SETTINGS")
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
            if auto_refresh:
                time.sleep(30)
                st.rerun()
    
    def _render_main_content(self):
        """Render the main content area."""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Data", "🤖 AI Predictions", "💼 Portfolio", "📊 Analytics"])
        
        with tab1:
            self._render_market_data_tab()
        
        with tab2:
            self._render_ai_predictions_tab()
        
        with tab3:
            self._render_portfolio_tab()
        
        with tab4:
            self._render_analytics_tab()
    
    def _render_market_data_tab(self):
        """Render market data tab."""
        st.header("📈 Real-Time Market Data")
        
        # Get data for selected symbol
        symbol = st.selectbox("Symbol", self.symbols, key="market_symbol")
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], key="market_period")
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Display current price
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
                with col3:
                    st.metric("Volume", f"{hist['Volume'].iloc[-1]:,}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=symbol
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("Historical Data")
                st.dataframe(hist.tail(10), use_container_width=True)
                
            else:
                st.error(f"No data available for {symbol}")
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
    
    def _render_ai_predictions_tab(self):
        """Render AI predictions tab."""
        st.header("🤖 AI Market Predictions")
        
        st.info("AI prediction features will be available in the full version with additional dependencies.")
        
        # Simple moving average prediction
        symbol = st.selectbox("Symbol", self.symbols, key="ai_symbol")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                # Calculate moving averages
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                hist['MA50'] = hist['Close'].rolling(window=50).mean()
                
                # Simple prediction based on trend
                current_price = hist['Close'].iloc[-1]
                ma20 = hist['MA20'].iloc[-1]
                ma50 = hist['MA50'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("20-day MA", f"${ma20:.2f}")
                with col3:
                    st.metric("50-day MA", f"${ma50:.2f}")
                
                # Trend analysis
                if current_price > ma20 > ma50:
                    st.success("🟢 BULLISH TREND: Price above both moving averages")
                elif current_price < ma20 < ma50:
                    st.error("🔴 BEARISH TREND: Price below both moving averages")
                else:
                    st.warning("🟡 MIXED SIGNALS: Conflicting trend indicators")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='white')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='20-day MA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='50-day MA', line=dict(color='red')))
                
                fig.update_layout(
                    title=f"{symbol} Technical Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in AI analysis: {str(e)}")
    
    def _render_portfolio_tab(self):
        """Render portfolio tab."""
        st.header("💼 Portfolio Management")
        
        st.info("Portfolio management features will be available in the full version.")
        
        # Simple portfolio simulation
        st.subheader("Portfolio Simulation")
        
        # Portfolio allocation
        allocations = {}
        total_allocation = 0
        
        for symbol in self.symbols[:5]:  # Use first 5 symbols
            allocation = st.slider(f"{symbol} Allocation (%)", 0, 100, 20, key=f"alloc_{symbol}")
            allocations[symbol] = allocation
            total_allocation += allocation
        
        if total_allocation != 100:
            st.warning(f"Total allocation: {total_allocation}%. Should equal 100%.")
        else:
            st.success("✅ Portfolio allocation complete!")
            
            # Calculate portfolio value
            portfolio_value = 100000  # $100k starting value
            portfolio_data = []
            
            for symbol, allocation in allocations.items():
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.info.get('regularMarketPrice', 0)
                    shares = (portfolio_value * allocation / 100) / current_price if current_price > 0 else 0
                    value = shares * current_price
                    
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Allocation': f"{allocation}%",
                        'Shares': f"{shares:.2f}",
                        'Current Price': f"${current_price:.2f}",
                        'Value': f"${value:.2f}"
                    })
                except:
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Allocation': f"{allocation}%",
                        'Shares': "N/A",
                        'Current Price': "N/A",
                        'Value': "N/A"
                    })
            
            st.subheader("Portfolio Summary")
            st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
    
    def _render_analytics_tab(self):
        """Render analytics tab."""
        st.header("📊 Market Analytics")
        
        st.info("Advanced analytics features will be available in the full version.")
        
        # Simple market overview
        st.subheader("Market Overview")
        
        market_data = []
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_data.append({
                    'Symbol': symbol,
                    'Price': f"${info.get('regularMarketPrice', 0):.2f}",
                    'Change': f"{info.get('regularMarketChangePercent', 0):.2f}%",
                    'Volume': f"{info.get('volume', 0):,}",
                    'Market Cap': f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A"
                })
            except:
                market_data.append({
                    'Symbol': symbol,
                    'Price': "N/A",
                    'Change': "N/A",
                    'Volume': "N/A",
                    'Market Cap': "N/A"
                })
        
        st.dataframe(pd.DataFrame(market_data), use_container_width=True)

def main():
    """Main application entry point."""
    terminal = MorganVuoksiTerminal()
    terminal.run()

if __name__ == "__main__":
    main() 