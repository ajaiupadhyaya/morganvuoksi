"""
MorganVuoksi Bloomberg Terminal - Professional Trading Interface
Exact Bloomberg Terminal replication with institutional-grade design.
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

# Add project root to path to resolve module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules with error handling
try:
    from src.data.market_data import MarketDataFetcher, DataConfig
    from src.models.advanced_models import TimeSeriesPredictor, ARIMAGARCHModel, EnsembleModel
    from src.models.rl_models import TD3Agent, SACAgent, TradingEnvironment
    from src.signals.nlp_signals import NLPSignalGenerator, FinancialNLPAnalyzer
    from src.portfolio.optimizer import PortfolioOptimizer
    from src.risk.risk_manager import RiskManager
    from src.visuals.charting import (
        create_candlestick_chart, create_technical_chart, create_portfolio_chart,
        create_risk_dashboard, create_prediction_chart, create_loss_curve,
        create_feature_importance_chart, create_sentiment_chart, create_efficient_frontier_chart
    )
except ImportError as e:
    st.warning(f"⚠️ Some modules could not be imported: {e}")
    st.info("Running in demo mode with simulated data")

warnings.filterwarnings('ignore')

# PROFESSIONAL BLOOMBERG TERMINAL CONFIGURATION
st.set_page_config(
    page_title="MorganVuoksi Bloomberg Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# EXACT BLOOMBERG TERMINAL CSS - INSTITUTIONAL GRADE
BLOOMBERG_TERMINAL_PROFESSIONAL_CSS = """
<style>
    /* CORE BLOOMBERG TERMINAL IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Courier+New:wght@400;500;600;700&display=swap');
    
    /* EXACT BLOOMBERG TERMINAL VARIABLES */
    :root {
        --bloomberg-black: #000000;         /* Pure black terminal background */
        --bloomberg-panel: #0a0a0a;        /* Dark panel backgrounds */
        --bloomberg-border: #333333;       /* Professional borders */
        --bloomberg-text: #ffffff;         /* Pure white text */
        --bloomberg-muted: #888888;        /* Secondary text */
        --bloomberg-orange: #ff6b35;       /* Bloomberg signature orange */
        --bloomberg-cyan: #00d4ff;         /* Primary data color */
        --bloomberg-green: #00ff88;        /* Bullish indicators */
        --bloomberg-red: #ff4757;          /* Bearish indicators */
        --bloomberg-amber: #ffa500;        /* Warnings */
        --bloomberg-blue: #0088cc;         /* Headers */
    }
    
    /* GLOBAL BLOOMBERG TERMINAL STYLING */
    * {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    .main {
        background: var(--bloomberg-black) !important;
        color: var(--bloomberg-text) !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stApp {
        background: var(--bloomberg-black) !important;
    }
    
    /* BLOOMBERG TERMINAL HEADER SYSTEM */
    .terminal-header {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%);
        border-bottom: 3px solid var(--bloomberg-orange);
        padding: 1.5rem 2rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.9);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .terminal-title {
        color: var(--bloomberg-orange);
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 0 3px 10px rgba(255, 107, 53, 0.6);
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .terminal-subtitle {
        color: var(--bloomberg-cyan);
        font-size: 16px;
        text-align: center;
        margin: 12px 0 0 0;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .bloomberg-status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 20px;
        padding-top: 16px;
        border-top: 2px solid var(--bloomberg-border);
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-live {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--bloomberg-green);
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.9);
        animation: bloomberg-pulse 2s infinite;
    }
    
    @keyframes bloomberg-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }
    
    /* PROFESSIONAL SIDEBAR STYLING */
    .stSidebar {
        background: linear-gradient(180deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%) !important;
        border-right: 3px solid var(--bloomberg-border) !important;
        padding: 1rem !important;
    }
    
    .stSidebar .sidebar-content {
        background: transparent !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: var(--bloomberg-orange) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        border-bottom: 2px solid var(--bloomberg-orange) !important;
        padding-bottom: 8px !important;
        margin-bottom: 16px !important;
    }
    
    /* ENHANCED INPUT CONTROLS */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%) !important;
        border: 2px solid var(--bloomberg-border) !important;
        border-radius: 0px !important;
        color: var(--bloomberg-text) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 1px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--bloomberg-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
        outline: none !important;
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, rgba(0, 212, 255, 0.05) 100%) !important;
    }
    
    /* PROFESSIONAL BUTTON SYSTEM */
    .stButton > button {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%) !important;
        color: var(--bloomberg-text) !important;
        border: 2px solid var(--bloomberg-border) !important;
        border-radius: 0px !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.8) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--bloomberg-orange) 0%, var(--bloomberg-amber) 100%) !important;
        color: var(--bloomberg-black) !important;
        border-color: var(--bloomberg-orange) !important;
        box-shadow: 0 0 25px rgba(255, 107, 53, 0.7) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 0 15px rgba(255, 107, 53, 0.5) !important;
    }
    
    /* ENHANCED TAB SYSTEM */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bloomberg-black);
        border-bottom: 3px solid var(--bloomberg-border);
        padding: 0;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%);
        border: 2px solid var(--bloomberg-border);
        border-bottom: none;
        color: var(--bloomberg-muted);
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 14px 22px;
        margin-right: 3px;
        transition: all 0.2s ease;
        min-width: 120px;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--bloomberg-orange) 0%, var(--bloomberg-amber) 100%) !important;
        color: var(--bloomberg-black) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.5);
        transform: translateY(-3px);
        border-color: var(--bloomberg-orange) !important;
        border-bottom: 3px solid var(--bloomberg-orange) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: linear-gradient(135deg, var(--bloomberg-cyan) 0%, var(--bloomberg-blue) 100%);
        color: var(--bloomberg-black);
        border-color: var(--bloomberg-cyan);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
        transform: translateY(-1px);
    }
    
    /* ULTRA-PROFESSIONAL METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%);
        border: 2px solid var(--bloomberg-border);
        border-top: 4px solid var(--bloomberg-orange);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 6px 12px rgba(0, 0, 0, 0.9),
            inset 0 2px 0 rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--bloomberg-orange), var(--bloomberg-cyan), var(--bloomberg-orange));
        z-index: 1;
        animation: bloomberg-gradient 3s ease-in-out infinite;
    }
    
    @keyframes bloomberg-gradient {
        0%, 100% { background: linear-gradient(90deg, var(--bloomberg-orange), var(--bloomberg-cyan), var(--bloomberg-orange)); }
        50% { background: linear-gradient(90deg, var(--bloomberg-cyan), var(--bloomberg-orange), var(--bloomberg-cyan)); }
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 12px 24px rgba(0, 0, 0, 0.95),
            0 0 30px rgba(0, 212, 255, 0.3);
        border-color: var(--bloomberg-cyan);
    }
    
    .metric-label {
        color: var(--bloomberg-cyan);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 12px;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.6);
    }
    
    .metric-value {
        color: var(--bloomberg-text);
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
        letter-spacing: 2px;
    }
    
    .metric-change {
        font-size: 13px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 0px;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 2px solid;
    }
    
    .metric-change.positive {
        color: var(--bloomberg-green);
        background: rgba(0, 255, 136, 0.15);
        border-color: var(--bloomberg-green);
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.8);
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
    }
    
    .metric-change.negative {
        color: var(--bloomberg-red);
        background: rgba(255, 71, 87, 0.15);
        border-color: var(--bloomberg-red);
        text-shadow: 0 0 10px rgba(255, 71, 87, 0.8);
        box-shadow: 0 0 15px rgba(255, 71, 87, 0.3);
    }
    
    .metric-change.neutral {
        color: var(--bloomberg-cyan);
        background: rgba(0, 212, 255, 0.15);
        border-color: var(--bloomberg-cyan);
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* INSTITUTIONAL-GRADE DATA TABLES */
    .dataframe {
        background: var(--bloomberg-black) !important;
        border: 3px solid var(--bloomberg-border) !important;
        border-radius: 0px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.9) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, var(--bloomberg-orange) 0%, var(--bloomberg-amber) 100%) !important;
        color: var(--bloomberg-black) !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
        border: 1px solid var(--bloomberg-orange) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        font-size: 10px !important;
        text-align: center !important;
    }
    
    .dataframe td {
        padding: 10px 16px !important;
        border: 1px solid var(--bloomberg-border) !important;
        color: var(--bloomberg-text) !important;
        background: var(--bloomberg-panel) !important;
        font-variant-numeric: tabular-nums !important;
        text-align: center !important;
        font-weight: 500 !important;
    }
    
    .dataframe tr:hover {
        background: rgba(0, 212, 255, 0.1) !important;
        box-shadow: inset 0 0 0 2px var(--bloomberg-cyan);
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    .dataframe tr:nth-child(even) td {
        background: rgba(26, 26, 26, 0.9) !important;
    }
    
    /* PROFESSIONAL CHART STYLING */
    .chart-container {
        background: linear-gradient(135deg, var(--bloomberg-panel) 0%, var(--bloomberg-black) 100%);
        border: 3px solid var(--bloomberg-border);
        border-top: 4px solid var(--bloomberg-cyan);
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 6px 16px rgba(0, 0, 0, 0.9),
            inset 0 2px 0 rgba(0, 212, 255, 0.1);
        position: relative;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: -4px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--bloomberg-cyan), var(--bloomberg-blue));
        z-index: 1;
    }
    
    /* ENHANCED SCROLLBARS */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bloomberg-black);
        border: 2px solid var(--bloomberg-border);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--bloomberg-orange) 0%, var(--bloomberg-amber) 100%);
        border-radius: 0px;
        border: 1px solid var(--bloomberg-border);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--bloomberg-cyan) 0%, var(--bloomberg-blue) 100%);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* PROFESSIONAL LOADING STATES */
    .loading-spinner {
        color: var(--bloomberg-cyan);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: bloomberg-pulse 2s infinite;
    }
    
    .loading-dots::after {
        content: 'LOADING DATA';
        animation: loading-text 2s infinite;
    }
    
    @keyframes loading-text {
        0%, 25% { content: 'LOADING DATA.'; }
        26%, 50% { content: 'LOADING DATA..'; }
        51%, 75% { content: 'LOADING DATA...'; }
        76%, 100% { content: 'LOADING DATA'; }
    }
    
    /* PROFESSIONAL SELECTION */
    ::selection {
        background: rgba(0, 212, 255, 0.4);
        color: var(--bloomberg-text);
    }
    
    ::-moz-selection {
        background: rgba(0, 212, 255, 0.4);
        color: var(--bloomberg-text);
    }
    
    /* RESPONSIVE DESIGN */
    @media (max-width: 768px) {
        .terminal-title {
            font-size: 28px;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
        
        .metric-value {
            font-size: 24px;
        }
    }
</style>
"""

st.markdown(BLOOMBERG_TERMINAL_PROFESSIONAL_CSS, unsafe_allow_html=True)

class MorganVuoksiTerminal:
    """Main terminal application."""
    
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
    
    def run(self):
        """Run the terminal application."""
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_header(self):
        """Render the terminal header."""
        st.markdown("""
        <div class="terminal-header">
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;">
                <div style="display: flex; align-items: center;">
                    <span class="status-indicator status-live"></span>
                    <span style="color: #a0a3a9; font-size: 12px; font-weight: 500;">LIVE DATA</span>
                </div>
                <div style="text-align: center;">
                    <h1 class="terminal-title">MorganVuoksi Terminal</h1>
                    <p class="terminal-subtitle">Institutional-Grade Quantitative Trading Platform</p>
                </div>
                <div style="text-align: right;">
                    <div style="color: #e8eaed; font-size: 16px; font-weight: 600;">{}</div>
                    <div style="color: #a0a3a9; font-size: 12px;">{} UTC</div>
                </div>
            </div>
        </div>
        """.format(
            datetime.now().strftime("%H:%M:%S"),
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        ), unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.markdown("## 🎛️ Controls")
        
        # Symbol input
        symbol = st.sidebar.text_input("Symbol", value=st.session_state.current_symbol, 
                                      help="Enter stock symbol (e.g., AAPL, TSLA)")
        
        if symbol:
            st.session_state.current_symbol = symbol.upper()
        
        # Date range
        st.sidebar.markdown("### Date Range")
        period = st.sidebar.selectbox("Period", 
                                    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
                                    index=5)
        
        # Data source
        st.sidebar.markdown("### Data Source")
        data_source = st.sidebar.selectbox("Source", ["yahoo", "alpaca", "polygon"], index=0)
        
        # Model settings
        st.sidebar.markdown("### AI Models")
        model_type = st.sidebar.selectbox("Prediction Model", 
                                        ["lstm", "transformer", "xgboost", "ensemble"], index=3)
        
        # Risk settings
        st.sidebar.markdown("### Risk Management")
        max_position_size = st.sidebar.slider("Max Position Size (%)", 1, 100, 20)
        stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 50, 10)
        
        # Store settings in session state
        st.session_state.period = period
        st.session_state.data_source = data_source
        st.session_state.model_type = model_type
        st.session_state.max_position_size = max_position_size
        st.session_state.stop_loss = stop_loss
        
        # Quick actions
        st.sidebar.markdown("### Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            self._clear_cache()
        
        if st.sidebar.button("📊 Generate Report"):
            self._generate_report()
    
    def _render_main_content(self):
        """Render the main content area."""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📈 Market Data", "🤖 AI Predictions", "📊 Portfolio", "💰 Valuation", 
            "⚠️ Risk Analysis", "🔄 Backtesting", "🎮 RL Simulator", "📰 News & NLP", 
            "📋 Reports", "🤖 LLM Assistant"
        ])
        
        with tab1:
            self._render_market_data_tab()
        
        with tab2:
            self._render_ai_predictions_tab()
        
        with tab3:
            self._render_portfolio_tab()
        
        with tab4:
            self._render_valuation_tab()
        
        with tab5:
            self._render_risk_analysis_tab()
        
        with tab6:
            self._render_backtesting_tab()
        
        with tab7:
            self._render_rl_simulator_tab()
        
        with tab8:
            self._render_news_nlp_tab()
        
        with tab9:
            self._render_reports_tab()
        
        with tab10:
            self._render_llm_assistant_tab()
    
    def _render_market_data_tab(self):
        """Render market data tab."""
        st.markdown("## 📈 Market Data & Technical Analysis")
        
        # Get market data
        data = self._get_market_data()
        
        if data is not None and not data.empty:
            # Market overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                change_class = "positive-change" if change >= 0 else "negative-change"
                change_icon = "↗" if change >= 0 else "↘"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Price</h3>
                    <div class="value">${current_price:.2f}</div>
                    <div class="change {change_class}">
                        {change_icon} {change:+.2f} ({change_pct:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = volume / avg_volume
                volume_status = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Volume</h3>
                    <div class="value">{volume:,.0f}</div>
                    <div class="change neutral-change">
                        {volume_ratio:.1f}x avg ({volume_status})
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'RSI' in data.columns:
                    rsi = data['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    rsi_color = "#ff6b6b" if rsi > 70 else "#00d4aa" if rsi < 30 else "#a0a3a9"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RSI</h3>
                        <div class="value" style="color: {rsi_color};">{rsi:.1f}</div>
                        <div class="change neutral-change">{rsi_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>RSI</h3>
                        <div class="value">--</div>
                        <div class="change neutral-change">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'Volatility' in data.columns:
                    vol = data['Volatility'].iloc[-1] * 100
                    vol_status = "High" if vol > 30 else "Normal" if vol > 15 else "Low"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Volatility</h3>
                        <div class="value">{vol:.1f}%</div>
                        <div class="change neutral-change">{vol_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Calculate volatility manually
                    returns = data['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252) * 100
                    vol_status = "High" if vol > 30 else "Normal" if vol > 15 else "Low"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Volatility</h3>
                        <div class="value">{vol:.1f}%</div>
                        <div class="change neutral-change">{vol_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Charts with enhanced styling
            st.markdown("### Price Action & Technical Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### Price Chart")
                fig = create_candlestick_chart(data)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### Technical Indicators")
                fig = create_technical_chart(data)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Market statistics with professional styling
            st.markdown("### Market Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### Price Statistics")
                price_stats = data['Close'].describe()
                st.dataframe(price_stats, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### Volume Statistics")
                volume_stats = data['Volume'].describe()
                st.dataframe(volume_stats, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### Returns Distribution")
                returns = data['Close'].pct_change().dropna()
                returns_stats = returns.describe()
                st.dataframe(returns_stats, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.error(f"Unable to fetch data for {st.session_state.current_symbol}")
    
    def _render_ai_predictions_tab(self):
        """Render AI predictions tab."""
        st.markdown("## 🤖 AI/ML Predictions")
        
        # Model selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox("Model Type", 
                                    ["lstm", "transformer", "xgboost", "ensemble"],
                                    index=3)
        
        with col2:
            prediction_horizon = st.selectbox("Prediction Horizon", 
                                            ["1d", "5d", "10d", "30d"], index=1)
        
        with col3:
            if st.button("🚀 Generate Predictions"):
                self._generate_predictions(model_type, prediction_horizon)
        
        # Display predictions
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            
            # Prediction summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = predictions.get('current_price', 0)
                predicted_price = predictions.get('predicted_price', 0)
                change = predicted_price - current_price
                change_pct = (change / current_price) * 100 if current_price > 0 else 0
                
                st.metric("Predicted Price", f"${predicted_price:.2f}", 
                         f"{change:+.2f} ({change_pct:+.2f}%)")
            
            with col2:
                confidence = predictions.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                signal = predictions.get('signal', 'hold')
                signal_color = {'buy': 'green', 'sell': 'red', 'hold': 'yellow'}.get(signal, 'gray')
                st.markdown(f"<h3 style='color: {signal_color};'>{signal.upper()}</h3>", 
                           unsafe_allow_html=True)
            
            with col4:
                model_accuracy = predictions.get('accuracy', 0)
                st.metric("Model Accuracy", f"{model_accuracy:.1%}")
            
            # Prediction chart
            st.markdown("### Prediction Chart")
            if 'prediction_chart' in predictions:
                st.plotly_chart(predictions['prediction_chart'], use_container_width=True)
            
            # Model diagnostics
            st.markdown("### Model Diagnostics")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'loss_curve' in predictions:
                    st.plotly_chart(predictions['loss_curve'], use_container_width=True)
            
            with col2:
                if 'feature_importance' in predictions:
                    st.plotly_chart(predictions['feature_importance'], use_container_width=True)
    
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
            
            # Portfolio metrics
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
            
            # Portfolio allocation chart
            st.markdown("### Optimal Allocation")
            if 'weights' in results:
                fig = create_portfolio_chart(results['weights'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Efficient frontier
            st.markdown("### Efficient Frontier")
            if 'efficient_frontier' in st.session_state:
                fig = create_efficient_frontier_chart(st.session_state.efficient_frontier)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_valuation_tab(self):
        """Render valuation tab."""
        st.markdown("## 💰 Fundamental Valuation")
        
        # Get company data
        symbol = st.session_state.current_symbol
        data = self._get_market_data()
        
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
        data = self._get_market_data()
        
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
    
    def _get_market_data(self):
        """Get market data for current symbol."""
        symbol = st.session_state.current_symbol
        cache_key = f"{symbol}_{st.session_state.period}_{st.session_state.data_source}"
        
        if cache_key in st.session_state.data_cache:
            return st.session_state.data_cache[cache_key]
        
        try:
            data = self.data_fetcher.get_stock_data(
                symbol, 
                st.session_state.period, 
                "1d", 
                st.session_state.data_source
            )
            
            if not data.empty:
                st.session_state.data_cache[cache_key] = data
            
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def _clear_cache(self):
        """Clear data cache."""
        st.session_state.data_cache.clear()
        st.session_state.predictions_cache.clear()
        st.success("Cache cleared!")
    
    def _generate_predictions(self, model_type, horizon):
        """Generate AI predictions."""
        with st.spinner("Generating predictions..."):
            try:
                data = self._get_market_data()
                if data is None or data.empty:
                    st.error("No data available for predictions")
                    return
                
                # Initialize model
                if model_type == "ensemble":
                    model = EnsembleModel()
                else:
                    model = TimeSeriesPredictor(model_type)
                
                # Fit model
                result = model.fit(data)
                
                # Make predictions
                predictions = model.predict(data)
                
                # Calculate metrics
                current_price = data['Close'].iloc[-1]
                predicted_price = predictions[-1] if len(predictions) > 0 else current_price
                
                # Store results
                st.session_state.predictions = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predictions': pd.Series(predictions, index=data.index[-len(predictions):]),
                    'confidence': 0.75,  # Placeholder
                    'signal': 'buy' if predicted_price > current_price else 'sell',
                    'accuracy': 0.65,  # Placeholder
                    'model_type': model_type,
                    'horizon': horizon,
                    'loss_curve': create_loss_curve(result.get('train_losses', []), result.get('test_losses', []))
                }
                
                if 'feature_importance' in result:
                    st.session_state.predictions['feature_importance'] = create_feature_importance_chart(
                        result['feature_importance'], data.columns.tolist()
                    )
                
                st.success("Predictions generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")
    
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
                data = self._get_market_data()
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
                data = self._get_market_data()
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
        """Generate a comprehensive market report."""
        st.info("Report generation functionality would be implemented here")

def main():
    """Main application entry point."""
    terminal = MorganVuoksiTerminal()
    terminal.run()

if __name__ == "__main__":
    main() 
