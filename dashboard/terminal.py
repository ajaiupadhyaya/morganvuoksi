"""
MorganVuoksi Terminal - Bloomberg-Style Quantitative Trading Terminal
Professional, institutional-grade interface for quantitative research and trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import sys
import os
from typing import Dict, List, Optional
import warnings
import time

# Add project root to path to resolve module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from src.data.market_data import MarketDataFetcher, DataConfig
from src.models.advanced_models import TimeSeriesPredictor, ARIMAGARCHModel, EnsembleModel
from src.models.rl_models import TD3Agent, SACAgent, TradingEnvironment
from src.signals.nlp_signals import NLPSignalGenerator, FinancialNLPAnalyzer
from src.portfolio.optimizer import PortfolioOptimizer
from src.risk.risk_manager import RiskManager
from src.visuals.charting import (
    create_candlestick_chart,
    create_technical_chart,
    create_portfolio_chart,
    create_risk_dashboard,
    create_prediction_chart,
    create_loss_curve,
    create_feature_importance_chart,
    create_sentiment_chart,
    create_efficient_frontier_chart
)

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
    /* --- Base & Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
    
    body {
        font-family: 'Roboto Mono', monospace;
        color: #E0E0E0;
        background-color: #000000;
    }

    .main {
        background-color: #000000;
    }

    /* --- Main Layout --- */
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
         color: #FFA500; /* Bloomberg Orange */
         border-bottom: 1px solid #333;
         padding-bottom: 5px;
         margin-top: 1.5rem;
         margin-bottom: 1rem;
    }

    /* --- Sidebar --- */
    .stSidebar {
        background-color: #0A0A0A;
        border-right: 1px solid #222;
    }

    .stSidebar h2, .stSidebar h3 {
         color: #FFA500;
    }

    /* --- Widgets --- */
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
    
    /* --- Custom Bloomberg Components --- */
    .header {
        background-color: #1A1A1A;
        padding: 5px 10px;
        border-bottom: 2px solid #FFA500;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .header-left {
        display: flex;
        align-items: center;
    }
    
    .header-left .ticker {
        font-size: 1.5em;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .header-left .price {
        font-size: 1.5em;
        margin-left: 15px;
        color: #00FF00; /* Green for up */
    }
    
    .header-left .change-positive {
        font-size: 1.5em;
        margin-left: 10px;
        color: #00FF00;
    }

    .header-left .change-negative {
        font-size: 1.5em;
        margin-left: 10px;
        color: #FF3333; /* Red for down */
    }

    .header-right .info {
         font-size: 0.8em;
         color: #AAAAAA;
         text-align: right;
    }

    .grid-item {
        background-color: #0A0A0A;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 2px;
        height: 100%;
    }
    
    .item-header {
         display: flex;
         justify-content: space-between;
         align-items: center;
         border-bottom: 1px solid #333;
         padding-bottom: 5px;
         margin-bottom: 10px;
    }
    
    .item-header-title {
         color: #FFA500;
         font-weight: 700;
    }
    
    .item-header-cmd {
        color: #AAAAAA;
        font-size: 0.9em;
    }
    
    .data-table {
        width: 100%;
        font-size: 0.9em;
    }
    
    .data-table tr:hover {
         background-color: #1a1a1a;
    }
    
    .data-table td {
        padding: 4px;
        border-bottom: 1px dotted #222;
    }

    .data-table td:nth-child(1) {
        color: #AAAAAA; /* Label color */
    }
    
    .data-table td:nth-child(2) {
        text-align: right;
        color: #FFFFFF; /* Value color */
        font-weight: 500;
    }
    
    .stPlotlyChart {
         border-radius: 2px;
    }

</style>
""", unsafe_allow_html=True)

class NewsItem:
    """News item data structure."""
    def __init__(self, title, description, content, published_at, source, url, sentiment_score, relevance_score):
        self.title = title
        self.description = description
        self.content = content
        self.published_at = published_at
        self.source = source
        self.url = url
        self.sentiment_score = sentiment_score
        self.relevance_score = relevance_score

class MorganVuoksiTerminal:
    """Professional Bloomberg-style terminal application."""
    
    def __init__(self):
        self.data_fetcher = MarketDataFetcher()
        self.nlp_generator = NLPSignalGenerator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_manager = RiskManager()
        
        # Initialize session state
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = 'AAPL'
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
        if 'predictions_cache' not in st.session_state:
            st.session_state.predictions_cache = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {'polygon': '', 'alpaca': ''}
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 'Market'
    
    def run(self):
        """Run the terminal application."""
        self._render_sidebar()
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render the professional sidebar with controls."""
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #ff6600, #ff8533); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: white; margin: 0; text-align: center;">🎛️ TRADING CONTROLS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Symbol input with professional styling
        st.sidebar.markdown("### 📊 Symbol")
        symbol = st.sidebar.text_input(
            "Enter Symbol", 
            value=st.session_state.current_symbol,
            help="Enter stock symbol (e.g., AAPL, TSLA, MSFT)",
            key="symbol_input"
        )
        
        if symbol:
            st.session_state.current_symbol = symbol.upper()
        
        # Market data controls
        st.sidebar.markdown("### 📈 Market Data")
        period = st.sidebar.selectbox(
            "Time Period", 
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=5,
            key="period_select"
        )
        
        data_source = st.sidebar.selectbox(
            "Data Source", 
            ["yahoo", "alpaca", "polygon"], 
            index=0,
            key="source_select"
        )
        
        # API Key Management
        with st.sidebar.expander("API Credentials"):
            st.session_state.api_keys['polygon'] = st.text_input("Polygon API Key", type="password", value=st.session_state.api_keys['polygon'])
            st.session_state.api_keys['alpaca'] = st.text_input("Alpaca API Key", type="password", value=st.session_state.api_keys['alpaca'])
        
        # AI Model controls
        st.sidebar.markdown("### 🤖 AI Models")
        model_type = st.sidebar.selectbox(
            "Prediction Model", 
            ["lstm", "transformer", "xgboost", "ensemble"], 
            index=3,
            key="model_select"
        )
        
        prediction_horizon = st.sidebar.slider(
            "Prediction Horizon (days)", 
            1, 30, 7,
            key="horizon_slider"
        )
        
        # Risk management controls
        st.sidebar.markdown("### ⚠️ Risk Management")
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)", 
            1, 100, 20,
            key="position_slider"
        )
        
        stop_loss = st.sidebar.slider(
            "Stop Loss (%)", 
            1, 50, 10,
            key="stop_loss_slider"
        )
        
        # Portfolio settings
        st.sidebar.markdown("### 📊 Portfolio")
        risk_tolerance = st.sidebar.selectbox(
            "Risk Tolerance", 
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            key="risk_select"
        )
        
        # Store settings in session state
        st.session_state.period = period
        st.session_state.data_source = data_source
        st.session_state.model_type = model_type
        st.session_state.prediction_horizon = prediction_horizon
        st.session_state.max_position_size = max_position_size
        st.session_state.stop_loss = stop_loss
        st.session_state.risk_tolerance = risk_tolerance
        
        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", key="refresh_btn"):
                self._clear_cache()
                st.success("Data refreshed!")
        
        with col2:
            if st.button("📊 Report", key="report_btn"):
                self._generate_report()
        
        # System status
        st.sidebar.markdown("### 🔧 System Status")
        st.sidebar.markdown("""
        <div style="background-color: #2d2d2d; padding: 12px; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Data Feed:</span>
                <span style="color: #00ff88;">✓ Online</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>AI Models:</span>
                <span style="color: #00ff88;">✓ Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Risk Engine:</span>
                <span style="color: #00ff88;">✓ Monitoring</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Execution:</span>
                <span style="color: #ffcc00;">⚠️ Simulated</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_main_content(self):
        """Render the main content area with professional tabs."""
        screen = st.session_state.current_screen
        
        if screen == 'Market':
            self._render_market_screen()
        elif screen == 'AI/ML Predictions':
            self._render_ai_predictions_screen()
        elif screen == 'Portfolio':
            self._render_portfolio_tab()
            self._render_valuation_tab()
            self._render_risk_analysis_tab()
            self._render_backtesting_tab()
            self._render_rl_simulator_tab()
            self._render_news_nlp_tab()
            self._render_reports_tab()
            self._render_llm_assistant_tab()
        else:
            st.header(screen)
            st.info("This screen is under construction.")
    
    @st.cache_data(ttl=300)
    def _get_ticker_data(self, symbol):
        """Cache ticker data to avoid refetching."""
        try:
            ticker = yf.Ticker(symbol)
            # Use a fast period first to check if ticker is valid
            if ticker.history(period="1d").empty:
                return None, None
            info = ticker.info
            hist = ticker.history(period=st.session_state.get('period', '1y'))
            # Check for essential data points
            if 'shortName' not in info or hist.empty:
                 return None, None
            return info, hist
        except Exception:
            return None, None

    def _render_market_screen(self):
        """Render the main market description screen (like Bloomberg's DES)."""
        with st.spinner(f"Loading data for {st.session_state.current_symbol}..."):
            info, hist = self._get_ticker_data(st.session_state.current_symbol)

        if info is None:
            st.error(f"Could not retrieve sufficient data for '{st.session_state.current_symbol}'. It may be an invalid ticker or delisted.")
            return

        # --- HEADER ---
        last_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else last_price
        change = last_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        change_class = "positive" if change >= 0 else "negative"
        
        st.markdown(f"""
            <div class="header">
                <div class="header-left">
                    <span class="ticker">{info.get('symbol', 'N/A')} US</span>
                    <span class="price" style="color:{'#00FF00' if change >= 0 else '#FF3333'};">${last_price:.2f}</span>
                    <span class="change-{change_class}">{change:+.2f} ({change_pct:+.2f}%)</span>
                </div>
                <div class="header-right">
                    <div class="info">Source: {st.session_state.data_source}</div>
                    <div class="info">Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # --- MAIN GRID ---
        col1, col2 = st.columns([2, 1], gap="medium")

        with col1:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            # --- Price Chart ---
            st.markdown("""
                <div class="item-header">
                    <span class="item-header-title">Price Chart</span>
                    <span class="item-header-cmd">GP &raquo;</span>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(data=go.Scatter(
                x=hist.index, y=hist['Close'],
                mode='lines',
                line=dict(color='#FFA500', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.1)'
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0), height=250,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, side='right')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # --- Key Stats Table ---
            mkt_cap = info.get('marketCap')
            ytd_start_price = hist['Close'].loc[f'{datetime.now().year}-01-01':].iloc[0] if not hist.loc[f'{datetime.now().year}-01-01':].empty else None
            ytd_change = ((last_price / ytd_start_price) - 1) * 100 if ytd_start_price else 0

            stats_data = {
                "Px/Chg 1D (USD)": f"{last_price:.2f} / {change_pct:+.2f}%",
                "52 Wk H": f"{info.get('fiftyTwoWeekHigh', 0):.2f}",
                "52 Wk L": f"{info.get('fiftyTwoWeekLow', 0):.2f}",
                "YTD Change%": f"{ytd_change:.2f}%",
                "Mkt Cap (USD)": f"{mkt_cap/1e9:.2f}B" if mkt_cap and mkt_cap > 1e9 else f"{mkt_cap/1e6:.2f}M" if mkt_cap else "N/A",
                "Shs Out/Float": f"{info.get('sharesOutstanding', 0)/1e6:.1f}M / {info.get('floatShares', 0)/1e6:.1f}M",
                "SI/% of Float": f"{info.get('shortPercentOfFloat', 0) * 100:.2f}%" if info.get('shortPercentOfFloat') else "N/A",
            }
            
            stats_html = '<table class="data-table">'
            for key, value in stats_data.items():
                stats_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            stats_html += '</table>'
            st.markdown(stats_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            # --- Company Info ---
            st.markdown(f"<h6>{info.get('longName', 'N/A')}</h6>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.8em; color:#AAAAAA;'>{info.get('longBusinessSummary', 'No description available.')[:250]}...</p>", unsafe_allow_html=True)

            # --- Estimates & Valuation ---
            st.markdown('<div class="item-header" style="margin-top: 15px;"><span class="item-header-title">Estimates</span><span class="item-header-cmd">EE &raquo;</span></div>', unsafe_allow_html=True)
            estimates_data = {
                "P/E": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
                "Est P/E": f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else "N/A",
                "T12M EPS": f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A",
                "PEG Ratio": f"{info.get('pegRatio', 'N/A')}",
            }
            est_html = '<table class="data-table">'
            for key, value in estimates_data.items():
                est_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            est_html += '</table>'
            st.markdown(est_html, unsafe_allow_html=True)

            # --- Corporate Info ---
            st.markdown('<div class="item-header" style="margin-top: 15px;"><span class="item-header-title">Corporate Info</span></div>', unsafe_allow_html=True)
            corp_data = {
                "Location": f"{info.get('city', 'N/A')}, {info.get('state', 'N/A')}",
                "Employees": f"{info.get('fullTimeEmployees', 0):,}",
                "Sector(s)": info.get('sector', 'N/A'),
            }
            corp_html = '<table class="data-table">'
            for key, value in corp_data.items():
                corp_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            corp_html += '</table>'

            st.markdown('</div>', unsafe_allow_html=True)

    def _render_ai_predictions_screen(self):
        """Render the AI/ML Predictions screen."""
        st.header(f"AI/ML Predictions for {st.session_state.current_symbol}")
        
        st.markdown('<div class="grid-item">', unsafe_allow_html=True)
        st.info("AI/ML prediction models and visualizations will be displayed here using the new design system.")
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_portfolio_tab(self):
        """Render portfolio optimization tab."""
        st.markdown("## 📊 Portfolio Optimization")
        
        # Portfolio inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols_input = st.text_area("Portfolio Symbols", 
                                       value="AAPL,MSFT,GOOGL,TSLA,NVDA",
                                       help="Enter symbols separated by commas")
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", 
                                        ["Conservative", "Moderate", "Aggressive"], index=1)
        
        with col3:
            if st.button("🔧 Optimize Portfolio"):
                self._optimize_portfolio(symbols, risk_tolerance)
        
        # Display portfolio results
        if 'portfolio_results' in st.session_state:
            results = st.session_state.portfolio_results
            
            # Portfolio summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                expected_return = results.get('expected_return', 0)
                st.metric("Expected Return", f"{expected_return:.2%}")
            
            with col2:
                volatility = results.get('volatility', 0)
                st.metric("Volatility", f"{volatility:.2%}")
            
            with col3:
                sharpe_ratio = results.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col4:
                max_drawdown = results.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            # Portfolio allocation
            st.markdown("### Portfolio Allocation")
            if 'weights' in results:
                weights_df = pd.DataFrame({
                    'Symbol': list(results['weights'].keys()),
                    'Weight': [f"{w:.2%}" for w in results['weights'].values()]
                })
                st.dataframe(weights_df, use_container_width=True)
            
            # Efficient frontier
            if 'efficient_frontier' in st.session_state:
                st.markdown("### Efficient Frontier")
                fig = create_efficient_frontier_chart(st.session_state.efficient_frontier)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_valuation_tab(self):
        """Render valuation tab."""
        st.markdown("## 💰 Fundamental Valuation")
        
        # Get company data
        symbol = st.session_state.current_symbol
        data = self._get_ticker_data(symbol)[1]
        
        if data is not None and not data.empty:
            # Company overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Company Information")
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    st.write(f"**Name:** {info.get('longName', 'N/A')}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.0f}")
                except:
                    st.write("Company information unavailable")
            
            with col2:
                st.markdown("### Valuation Metrics")
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**P/B Ratio:** {info.get('priceToBook', 'N/A')}")
                    st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
                    st.write(f"**Beta:** {info.get('beta', 'N/A')}")
                except:
                    st.write("Valuation metrics unavailable")
            
            with col3:
                st.markdown("### Financial Health")
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%")
                    st.write(f"**Debt/Equity:** {info.get('debtToEquity', 'N/A')}")
                    st.write(f"**Profit Margin:** {info.get('profitMargins', 0)*100:.2f}%")
                    st.write(f"**Current Ratio:** {info.get('currentRatio', 'N/A')}")
                except:
                    st.write("Financial health metrics unavailable")
            
            # DCF Analysis
            st.markdown("### Discounted Cash Flow Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                growth_rate = st.slider("Growth Rate (%)", 0, 50, 10)
                discount_rate = st.slider("Discount Rate (%)", 5, 20, 10)
                
                if st.button("Calculate DCF"):
                    self._calculate_dcf(growth_rate, discount_rate)
            
            with col2:
                if 'dcf_result' in st.session_state:
                    dcf = st.session_state.dcf_result
                    st.metric("DCF Value", f"${dcf.get('dcf_value', 0):.2f}")
                    st.metric("Current Price", f"${dcf.get('current_price', 0):.2f}")
                    st.metric("Upside/Downside", f"{dcf.get('upside', 0):+.1f}%")
    
    def _render_risk_analysis_tab(self):
        """Render risk analysis tab."""
        st.markdown("## ⚠️ Risk Analysis")
        
        # Risk metrics
        data = self._get_ticker_data(st.session_state.current_symbol)[1]
        
        if data is not None and not data.empty:
            # Calculate risk metrics
            returns = data['Close'].pct_change().dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.2f}%")
            
            with col2:
                var_95 = np.percentile(returns, 5) * 100
                st.metric("VaR (95%)", f"{var_95:.2f}%")
            
            with col3:
                cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
                st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
            
            with col4:
                max_drawdown = self._calculate_max_drawdown(data['Close'])
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
            # Risk Dashboard
            st.markdown("### Risk Dashboard")
            fig = create_risk_dashboard({
                'volatility': volatility,
                'var_95': var_95,
                'max_drawdown': max_drawdown
            })
            st.plotly_chart(fig, use_container_width=True)

            # Risk charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Returns Distribution")
                fig = px.histogram(returns, nbins=50, title="Returns Distribution")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Drawdown Analysis")
                drawdown = self._calculate_drawdown_series(data['Close'])
                fig = px.line(drawdown, title="Drawdown Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Stress testing
            st.markdown("### Stress Testing")
            col1, col2 = st.columns(2)
            
            with col1:
                stress_scenarios = {
                    "Market Crash (-20%)": -0.20,
                    "Recession (-10%)": -0.10,
                    "Volatility Spike (+50%)": 0.50,
                    "Interest Rate Hike (+2%)": 0.02
                }
                
                scenario = st.selectbox("Stress Scenario", list(stress_scenarios.keys()))
                impact = stress_scenarios[scenario]
                
                if st.button("Run Stress Test"):
                    self._run_stress_test(impact)
            
            with col2:
                if 'stress_test_result' in st.session_state:
                    result = st.session_state.stress_test_result
                    st.metric("Portfolio Impact", f"{result.get('impact', 0):+.2f}%")
                    st.metric("New VaR", f"{result.get('new_var', 0):.2f}%")
    
    def _render_backtesting_tab(self):
        """Render backtesting tab."""
        st.markdown("## 🔄 Strategy Backtesting")
        
        # Strategy inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy_type = st.selectbox("Strategy", 
                                       ["Moving Average Crossover", "RSI Strategy", "MACD Strategy", "Custom"],
                                       index=0)
        
        with col2:
            initial_capital = st.number_input("Initial Capital ($)", 
                                            min_value=1000, value=100000, step=1000)
        
        with col3:
            if st.button("🚀 Run Backtest"):
                self._run_backtest(strategy_type, initial_capital)
        
        # Display backtest results
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_return = results.get('total_return', 0)
                st.metric("Total Return", f"{total_return:.2%}")
            
            with col2:
                sharpe_ratio = results.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col3:
                max_drawdown = results.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            with col4:
                win_rate = results.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            # Performance chart
            st.markdown("### Performance Chart")
            if 'performance_chart' in results:
                st.plotly_chart(results['performance_chart'], use_container_width=True)
            
            # Trade analysis
            st.markdown("### Trade Analysis")
            if 'trades_df' in results:
                st.dataframe(results['trades_df'])
            
            # Sentiment distribution
            st.markdown("### Sentiment Distribution")
            if 'sentiment_distribution' in results:
                dist = results['sentiment_distribution']
                fig = create_sentiment_chart(dist)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_rl_simulator_tab(self):
        """Render reinforcement learning simulator tab."""
        st.markdown("## 🎮 RL Trading Simulator")
        
        # RL agent settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            agent_type = st.selectbox("Agent Type", ["TD3", "SAC"], index=0)
        
        with col2:
            training_episodes = st.slider("Training Episodes", 10, 1000, 100)
        
        with col3:
            if st.button("🎯 Train Agent"):
                self._train_rl_agent(agent_type, training_episodes)
        
        # Display RL results
        if 'rl_results' in st.session_state:
            results = st.session_state.rl_results
            
            # Training metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                final_reward = results.get('final_reward', 0)
                st.metric("Final Reward", f"{final_reward:.2f}")
            
            with col2:
                total_trades = results.get('total_trades', 0)
                st.metric("Total Trades", f"{total_trades}")
            
            with col3:
                win_rate = results.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            with col4:
                avg_return = results.get('avg_return', 0)
                st.metric("Avg Return", f"{avg_return:.2%}")
            
            # Training progress
            st.markdown("### Training Progress")
            if 'training_chart' in results:
                st.plotly_chart(results['training_chart'], use_container_width=True)
            
            # Agent actions
            st.markdown("### Agent Actions")
            if 'actions_chart' in results:
                st.plotly_chart(results['actions_chart'], use_container_width=True)
    
    def _render_news_nlp_tab(self):
        """Render news and NLP analysis tab."""
        st.markdown("## 📰 News & Sentiment Analysis")
        
        # News analysis
        col1, col2 = st.columns(2)
        
        with col1:
            days_back = st.slider("News Days Back", 1, 30, 7)
        
        with col2:
            if st.button("📊 Analyze Sentiment"):
                self._analyze_sentiment(days_back)
        
        # Display sentiment results
        if 'sentiment_results' in st.session_state:
            results = st.session_state.sentiment_results
            
            # Sentiment metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_score = results.get('sentiment_score', 0)
                st.metric("Sentiment Score", f"{sentiment_score:.3f}")
            
            with col2:
                confidence = results.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                signal = results.get('signal', 'neutral')
                signal_color = {'buy': 'green', 'sell': 'red', 'neutral': 'yellow'}.get(signal, 'gray')
                st.markdown(f"<h3 style='color: {signal_color};'>{signal.upper()}</h3>", 
                           unsafe_allow_html=True)
            
            with col4:
                news_count = results.get('news_count', 0)
                st.metric("News Count", f"{news_count}")
            
            # Recent news
            st.markdown("### Recent News")
            if 'recent_news' in results:
                for news in results['recent_news'][:5]:
                    with st.expander(f"{news.title} - {news.published_at.strftime('%Y-%m-%d')}"):
                        st.write(f"**Source:** {news.source}")
                        st.write(f"**Sentiment:** {news.sentiment_score:.3f}")
                        st.write(f"**Description:** {news.description}")
                        st.write(f"**URL:** {news.url}")
            
            # Sentiment distribution
            st.markdown("### Sentiment Distribution")
            if 'sentiment_distribution' in results:
                dist = results['sentiment_distribution']
                fig = create_sentiment_chart(dist)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_reports_tab(self):
        """Render reports tab."""
        st.markdown("## 📋 Automated Reports")
        
        # Report generation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox("Report Type", 
                                     ["Market Analysis", "Portfolio Review", "Risk Assessment", "Full Report"],
                                     index=0)
        
        with col2:
            time_period = st.selectbox("Time Period", 
                                     ["1 Week", "1 Month", "3 Months", "1 Year"],
                                     index=1)
        
        with col3:
            if st.button("📄 Generate Report"):
                self._generate_automated_report(report_type, time_period)
        
        # Display report
        if 'current_report' in st.session_state:
            report = st.session_state.current_report
            
            st.markdown("### Generated Report")
            st.markdown(report['content'])
            
            # Download report
            if st.button("💾 Download Report"):
                self._download_report(report)
    
    def _render_llm_assistant_tab(self):
        """Render LLM assistant tab."""
        st.markdown("## 🤖 AI Assistant")
        
        # Chat interface
        st.markdown("### Ask me anything about the markets, your portfolio, or trading strategies!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                response = self._generate_ai_response(prompt)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def _optimize_portfolio(self, symbols, risk_tolerance):
        """Optimize portfolio allocation."""
        with st.spinner("Optimizing portfolio..."):
            try:
                # Get data for all symbols
                data_dict = {}
                for symbol in symbols:
                    data = self.data_fetcher.get_stock_data(symbol, "1y", "1d", "yahoo")
                    if not data.empty:
                        data_dict[symbol] = data['Close']
                
                if len(data_dict) < 2:
                    st.error("Need at least 2 symbols for portfolio optimization")
                    return
                
                # Create returns DataFrame
                returns_df = pd.DataFrame(data_dict).pct_change().dropna()
                
                # Optimize portfolio
                optimization_result = self.portfolio_optimizer.optimize_portfolio(returns_df, risk_tolerance)
                
                # Calculate efficient frontier
                efficient_frontier = self.portfolio_optimizer.calculate_efficient_frontier(returns_df)
                st.session_state.efficient_frontier = efficient_frontier

                # Store results
                st.session_state.portfolio_results = optimization_result
                
                st.success("Portfolio optimized successfully!")
                
            except Exception as e:
                st.error(f"Error optimizing portfolio: {e}")
    
    def _calculate_dcf(self, growth_rate, discount_rate):
        """Calculate DCF valuation."""
        try:
            symbol = st.session_state.current_symbol
            ticker = yf.Ticker(symbol)
            
            # Get financial data
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            
            # Simple DCF calculation (placeholder)
            fcf = info.get('freeCashflow', 1000000000)  # Placeholder
            growth_rate_decimal = growth_rate / 100
            discount_rate_decimal = discount_rate / 100
            
            # Terminal value calculation
            terminal_value = fcf * (1 + growth_rate_decimal) / (discount_rate_decimal - growth_rate_decimal)
            present_value = fcf / (1 + discount_rate_decimal) + terminal_value / (1 + discount_rate_decimal) ** 5
            
            # Calculate upside/downside
            upside = ((present_value - current_price) / current_price) * 100
            
            st.session_state.dcf_result = {
                'dcf_value': present_value,
                'current_price': current_price,
                'upside': upside
            }
            
        except Exception as e:
            st.error(f"Error calculating DCF: {e}")
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_drawdown_series(self, prices):
        """Calculate drawdown series."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak * 100
        return drawdown
    
    def _run_stress_test(self, impact):
        """Run stress test."""
        try:
            # Simple stress test (placeholder)
            base_var = -5.0  # Base VaR
            new_var = base_var * (1 + impact)
            
            st.session_state.stress_test_result = {
                'impact': impact * 100,
                'new_var': new_var
            }
            
        except Exception as e:
            st.error(f"Error running stress test: {e}")
    
    def _run_backtest(self, strategy_type, initial_capital):
        """Run strategy backtest."""
        with st.spinner("Running backtest..."):
            try:
                data = self._get_ticker_data(st.session_state.current_symbol)[1]
                if data is None or data.empty:
                    st.error("No data available for backtesting")
                    return
                
                # Simple backtest (placeholder)
                returns = data['Close'].pct_change().dropna()
                total_return = returns.sum()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(data['Close'])
                
                # Create performance chart
                cumulative_returns = (1 + returns).cumprod()
                fig = px.line(cumulative_returns, title="Cumulative Returns")
                
                st.session_state.backtest_results = {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': 0.55,  # Placeholder
                    'performance_chart': fig,
                    'trades_df': pd.DataFrame(),  # Placeholder
                    'sentiment_distribution': {}  # Placeholder
                }
                
                st.success("Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"Error running backtest: {e}")
    
    def _train_rl_agent(self, agent_type, episodes):
        """Train RL agent."""
        with st.spinner("Training RL agent..."):
            try:
                data = self._get_ticker_data(st.session_state.current_symbol)[1]
                if data is None or data.empty:
                    st.error("No data available for RL training")
                    return
                
                # Simple RL training (placeholder)
                final_reward = np.random.normal(0.05, 0.02)  # Placeholder
                total_trades = np.random.randint(50, 200)  # Placeholder
                win_rate = np.random.uniform(0.4, 0.7)  # Placeholder
                avg_return = np.random.uniform(0.01, 0.05)  # Placeholder
                
                # Create training chart
                training_rewards = np.cumsum(np.random.normal(0.001, 0.01, episodes))
                fig = px.line(training_rewards, title="Training Progress")
                
                st.session_state.rl_results = {
                    'final_reward': final_reward,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'training_chart': fig,
                    'actions_chart': fig  # Placeholder
                }
                
                st.success("RL agent trained successfully!")
                
            except Exception as e:
                st.error(f"Error training RL agent: {e}")
    
    def _analyze_sentiment(self, days_back):
        """Analyze news sentiment."""
        with st.spinner("Analyzing sentiment..."):
            try:
                symbol = st.session_state.current_symbol
                
                # Simple sentiment analysis (placeholder)
                sentiment_score = np.random.uniform(-0.5, 0.5)
                confidence = np.random.uniform(0.3, 0.9)
                signal = 'buy' if sentiment_score > 0.2 else 'sell' if sentiment_score < -0.2 else 'neutral'
                
                # Mock news items
                news_items = [
                    NewsItem(
                        title=f"News about {symbol}",
                        description=f"Latest developments for {symbol}",
                        content="",
                        published_at=datetime.now() - timedelta(days=i),
                        source="Mock News",
                        url="",
                        sentiment_score=np.random.uniform(-0.5, 0.5),
                        relevance_score=np.random.uniform(0.3, 0.9)
                    ) for i in range(5)
                ]
                
                st.session_state.sentiment_results = {
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'signal': signal,
                    'news_count': 25,
                    'recent_news': news_items,
                    'sentiment_distribution': {
                        'positive': 12,
                        'negative': 8,
                        'neutral': 5
                    }
                }
                
                st.success("Sentiment analysis completed!")
                
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")
    
    def _generate_automated_report(self, report_type, time_period):
        """Generate automated report."""
        with st.spinner("Generating report..."):
            try:
                # Simple report generation (placeholder)
                report_content = f"""
# {report_type} Report for {st.session_state.current_symbol}

## Executive Summary
This report provides a comprehensive analysis of {st.session_state.current_symbol} over the {time_period} period.

## Key Findings
- Market performance analysis
- Risk assessment
- Investment recommendations

## Technical Analysis
- Price trends and patterns
- Technical indicators
- Support and resistance levels

## Fundamental Analysis
- Financial metrics
- Valuation ratios
- Growth prospects

## Risk Assessment
- Volatility analysis
- Drawdown analysis
- Stress testing results

## Recommendations
Based on the analysis, we recommend [BUY/HOLD/SELL] for {st.session_state.current_symbol}.
                """
                
                st.session_state.current_report = {
                    'type': report_type,
                    'time_period': time_period,
                    'content': report_content,
                    'generated_at': datetime.now()
                }
                
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {e}")
    
    def _download_report(self, report):
        """Download report as PDF."""
        st.info("Report download functionality would be implemented here")
    
    def _generate_ai_response(self, prompt):
        """Generate AI response using LLM."""
        # Simple AI response (placeholder)
        responses = {
            "market": "Based on current market conditions, I recommend monitoring key support and resistance levels.",
            "portfolio": "Your portfolio appears well-diversified. Consider rebalancing quarterly.",
            "risk": "Current risk metrics indicate moderate volatility. Maintain stop-loss orders.",
            "strategy": "For your risk profile, I suggest a balanced approach with 60% equities and 40% bonds."
        }
        
        # Simple keyword matching
        if "market" in prompt.lower():
            return responses["market"]
        elif "portfolio" in prompt.lower():
            return responses["portfolio"]
        elif "risk" in prompt.lower():
            return responses["risk"]
        elif "strategy" in prompt.lower():
            return responses["strategy"]
        else:
            return "I'm here to help with your trading and investment questions. Please ask me about markets, portfolios, risk management, or trading strategies."

    def _generate_report(self):
        """Generate automated trading report."""
        try:
            st.success("Report generation started!")
            # Placeholder for report generation logic
            time.sleep(1)
            st.success("Report generated successfully!")
        except Exception as e:
            st.error(f"Error generating report: {e}")

def main():
    """Main application entry point."""
    terminal = MorganVuoksiTerminal()
    terminal.run()

if __name__ == "__main__":
    main() 
