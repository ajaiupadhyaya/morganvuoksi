#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal - Bloomberg-Style Quantitative Finance Platform
Production-grade terminal with real-time data, AI predictions, and institutional features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import time
import logging
import os
import sys
from typing import Dict, List, Optional, Any
import warnings
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MorganVuoksi Elite Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "MorganVuoksi Elite Terminal - Bloomberg-Grade Quantitative Finance Platform"
    }
)

# Bloomberg-style CSS
BLOOMBERG_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%);
        color: #e8eaed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .terminal-header {
        background: linear-gradient(135deg, #1e2330 0%, #2a3142 100%);
        border: 1px solid #3a4152;
        border-radius: 12px;
        padding: 24px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .terminal-title {
        color: #00d4aa;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
        letter-spacing: 1px;
    }
    
    .terminal-subtitle {
        color: #a0a3a9;
        font-size: 14px;
        text-align: center;
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    
    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #3a4152;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-live {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00d4aa;
        box-shadow: 0 0 8px rgba(0, 212, 170, 0.6);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #1e2330 0%, #2a3142 100%);
        border-right: 1px solid #3a4152;
    }
    
    .stSidebar .stSelectbox > div > div {
        background: #2a3142;
        border: 1px solid #3a4152;
        color: #e8eaed;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2a3142 0%, #1e2330 100%);
        border: 1px solid #3a4152;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #0066cc);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        border-color: #00d4aa;
    }
    
    .metric-label {
        color: #a0a3a9;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #e8eaed;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-change {
        font-size: 12px;
        font-weight: 500;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .metric-change.positive {
        color: #00d4aa;
        background: rgba(0, 212, 170, 0.1);
    }
    
    .metric-change.negative {
        color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    .metric-change.neutral {
        color: #a0a3a9;
        background: rgba(160, 163, 169, 0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e2330;
        border-radius: 12px 12px 0 0;
        border: 1px solid #3a4152;
        border-bottom: none;
        padding: 8px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a0a3a9;
        font-weight: 500;
        font-size: 14px;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4aa, #0066cc);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(0, 212, 170, 0.1);
        color: #00d4aa;
    }
    
    /* Tab content */
    .stTabs [data-baseweb="tab-panel"] {
        background: #2a3142;
        border: 1px solid #3a4152;
        border-radius: 0 0 12px 12px;
        padding: 24px;
        min-height: 600px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #00d4aa 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 102, 204, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background: #2a3142;
        border: 1px solid #3a4152;
        border-radius: 8px;
        color: #e8eaed;
        font-size: 14px;
        padding: 12px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00d4aa;
        box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
    }
    
    /* Data tables */
    .dataframe {
        background: #2a3142;
        border: 1px solid #3a4152;
        border-radius: 12px;
        overflow: hidden;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .dataframe th {
        background: #1e2330;
        color: #e8eaed;
        font-weight: 600;
        padding: 16px;
        border-bottom: 2px solid #3a4152;
    }
    
    .dataframe td {
        padding: 12px 16px;
        border-bottom: 1px solid #3a4152;
        color: #a0a3a9;
    }
    
    .dataframe tr:hover {
        background: rgba(0, 212, 170, 0.05);
    }
    
    /* Chart containers */
    .chart-container {
        background: #2a3142;
        border: 1px solid #3a4152;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #00d4aa transparent transparent transparent;
    }
    
    /* Alert styling */
    .stAlert {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid #00d4aa;
        border-radius: 8px;
        color: #e8eaed;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4aa, #0066cc);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2330;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3a4152;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4a5568;
    }
</style>
"""

st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)

class EliteTerminal:
    """Bloomberg-grade Elite Terminal implementation."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'current_symbol': 'AAPL',
            'timeframe': '1Y',
            'auto_refresh': True,
            'refresh_interval': 30,
            'selected_model': 'ensemble',
            'prediction_horizon': 30,
            'portfolio_symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'watchlist': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META'],
            'data_cache': {},
            'last_refresh': datetime.now()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point."""
        try:
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
            self._handle_auto_refresh()
        except Exception as e:
            st.error(f"Terminal Error: {str(e)}")
            logger.error(f"Terminal error: {str(e)}")
    
    def _render_header(self):
        """Render professional header."""
        current_time = datetime.now().strftime("%H:%M:%S EST")
        utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Check API status
        api_status = self._check_api_status()
        
        header_html = f"""
        <div class="terminal-header">
            <div class="terminal-title">
                MORGANVUOKSI ELITE TERMINAL
            </div>
            <div class="terminal-subtitle">
                Institutional-Grade Quantitative Finance Platform
            </div>
            <div class="status-bar">
                <div class="status-indicator">
                    <div class="status-live"></div>
                    <span>LIVE DATA • {api_status}</span>
                </div>
                <div style="color: #a0a3a9; font-family: 'JetBrains Mono', monospace;">
                    {current_time} • {utc_time}
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def _check_api_status(self):
        """Check API server status."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=2)
            if response.status_code == 200:
                return "OPERATIONAL"
            else:
                return "LIMITED"
        except:
            return "OFFLINE"
    
    def _render_sidebar(self):
        """Render Bloomberg-style sidebar."""
        with st.sidebar:
            st.markdown("## 🎛️ TERMINAL CONTROLS", unsafe_allow_html=True)
            
            # Symbol input with quick select
            col1, col2 = st.columns([3, 1])
            with col1:
                symbol = st.text_input(
                    "SYMBOL",
                    value=st.session_state.current_symbol,
                    max_chars=10,
                    help="Enter stock symbol"
                ).upper()
            
            with col2:
                if st.button("📊", help="Analyze"):
                    st.session_state.current_symbol = symbol
                    st.experimental_rerun()
            
            # Quick symbol buttons
            st.markdown("**QUICK SELECT**")
            cols = st.columns(3)
            for i, sym in enumerate(st.session_state.watchlist[:6]):
                col = cols[i % 3]
                if col.button(sym, key=f"quick_{sym}", use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.experimental_rerun()
            
            st.markdown("---")
            
            # Market data settings
            st.markdown("### 📈 MARKET DATA")
            timeframe = st.selectbox(
                "TIMEFRAME",
                ['1D', '5D', '1M', '3M', '6M', '1Y', '2Y', '5Y'],
                index=5
            )
            st.session_state.timeframe = timeframe
            
            show_volume = st.checkbox("Show Volume", value=True)
            show_indicators = st.multiselect(
                "Technical Indicators",
                ['RSI', 'MACD', 'Bollinger Bands', 'SMA', 'EMA'],
                default=['RSI', 'MACD']
            )
            
            st.markdown("---")
            
            # AI/ML settings
            st.markdown("### 🤖 AI MODELS")
            model_type = st.selectbox(
                "MODEL TYPE",
                ['lstm', 'transformer', 'xgboost', 'ensemble'],
                index=3
            )
            st.session_state.selected_model = model_type
            
            horizon = st.slider(
                "PREDICTION HORIZON (DAYS)",
                min_value=1, max_value=90, value=30
            )
            st.session_state.prediction_horizon = horizon
            
            st.markdown("---")
            
            # System controls
            st.markdown("### ⚙️ SYSTEM")
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            st.session_state.auto_refresh = auto_refresh
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (sec)",
                    min_value=10, max_value=300, value=30
                )
                st.session_state.refresh_interval = refresh_interval
            
            col1, col2 = st.columns(2)
            if col1.button("🔄 REFRESH", use_container_width=True):
                self._clear_cache()
                st.experimental_rerun()
            
            if col2.button("⚙️ RESET", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()
    
    def _render_main_content(self):
        """Render main tabbed content."""
        tabs = st.tabs([
            "📈 MARKET DATA",
            "🤖 AI PREDICTIONS", 
            "📊 PORTFOLIO",
            "💰 VALUATION",
            "⚠️ RISK ANALYSIS",
            "🔄 BACKTESTING",
            "🎮 RL AGENTS",
            "📰 NEWS & NLP",
            "📋 REPORTS",
            "🤖 LLM ASSISTANT"
        ])
        
        with tabs[0]:
            self._render_market_data_tab()
        with tabs[1]:
            self._render_ai_predictions_tab()
        with tabs[2]:
            self._render_portfolio_tab()
        with tabs[3]:
            self._render_valuation_tab()
        with tabs[4]:
            self._render_risk_analysis_tab()
        with tabs[5]:
            self._render_backtesting_tab()
        with tabs[6]:
            self._render_rl_agents_tab()
        with tabs[7]:
            self._render_news_nlp_tab()
        with tabs[8]:
            self._render_reports_tab()
        with tabs[9]:
            self._render_llm_assistant_tab()
    
    def _render_market_data_tab(self):
        """Render enhanced market data tab."""
        st.markdown("## 📈 REAL-TIME MARKET DATA & ANALYSIS")
        
        # Get market data
        symbol = st.session_state.current_symbol
        market_data = self._get_market_data(symbol)
        
        if market_data:
            # Market overview metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            symbol_info = market_data.get('symbol', {})
            current_price = symbol_info.get('price', 0)
            change_val = symbol_info.get('change_val', 0)
            change_pct = symbol_info.get('change_pct', 0)
            
            # Price metric
            change_class = "positive" if change_pct >= 0 else "negative"
            change_arrow = "↗" if change_pct >= 0 else "↘"
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PRICE</div>
                    <div class="metric-value">${current_price:.2f}</div>
                    <div class="metric-change {change_class}">
                        {change_arrow} ${change_val:+.2f} ({change_pct:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">VOLUME</div>
                    <div class="metric-value">{symbol_info.get('volume', '0')}</div>
                    <div class="metric-change neutral">24H Volume</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MARKET CAP</div>
                    <div class="metric-value">{symbol_info.get('market_cap', 'N/A')}</div>
                    <div class="metric-change neutral">Market Value</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pe_ratio = symbol_info.get('pe_ratio', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P/E RATIO</div>
                    <div class="metric-value">{pe_ratio}</div>
                    <div class="metric-change neutral">Valuation</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                beta = symbol_info.get('beta', 1.0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">BETA</div>
                    <div class="metric-value">{beta:.2f}</div>
                    <div class="metric-change neutral">Market Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive price chart
            self._render_interactive_price_chart(market_data)
            
            # Market data grid and watchlist
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._render_market_grid(market_data)
            
            with col2:
                self._render_enhanced_watchlist()
    
    def _render_interactive_price_chart(self, market_data):
        """Render interactive price chart with technical indicators."""
        historical_data = market_data.get('historical_data', {})
        dates = historical_data.get('dates', [])
        prices = historical_data.get('prices', {})
        tech_indicators = market_data.get('technical_indicators', {})
        
        if not dates or not prices:
            st.warning("No historical data available")
            return
        
        # Create candlestick chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price & Volume', 'RSI', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=prices.get('open', []),
                high=prices.get('high', []),
                low=prices.get('low', []),
                close=prices.get('close', []),
                name='Price',
                increasing_line_color='#00d4aa',
                decreasing_line_color='#ff6b6b'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['#00d4aa' if c >= o else '#ff6b6b' 
                 for o, c in zip(prices.get('open', []), prices.get('close', []))]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=prices.get('volume', []),
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # RSI
        if tech_indicators.get('rsi'):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=tech_indicators['rsi'],
                    name='RSI',
                    line=dict(color='#0066cc', width=2)
                ),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", row=2, col=1)
        
        # MACD
        if tech_indicators.get('macd') and tech_indicators.get('macd_signal'):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=tech_indicators['macd'],
                    name='MACD',
                    line=dict(color='#0066cc', width=2)
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=tech_indicators['macd_signal'],
                    name='Signal',
                    line=dict(color='#00d4aa', width=2)
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{st.session_state.current_symbol} - Advanced Price Analysis",
            template='plotly_dark',
            paper_bgcolor='rgba(42, 49, 66, 0.8)',
            plot_bgcolor='rgba(30, 35, 48, 0.8)',
            font=dict(color='#e8eaed'),
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Add volume axis
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data from API with caching."""
        cache_key = f"market_{symbol}_{st.session_state.timeframe}"
        
        # Check cache
        if cache_key in st.session_state.data_cache:
            cache_time, data = st.session_state.data_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=5):
                return data
        
        try:
            with st.spinner("Fetching market data..."):
                response = requests.get(
                    f"{self.api_base_url}/api/v1/terminal_data/{symbol}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.data_cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    st.error(f"API Error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            st.error("Unable to fetch market data. Please check API connection.")
            return None
    
    def _render_ai_predictions_tab(self):
        """Render AI predictions tab with advanced models."""
        st.markdown("## 🤖 AI PRICE PREDICTIONS & MODEL ANALYSIS")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("🚀 GENERATE PREDICTIONS", use_container_width=True):
                self._generate_predictions()
        
        with col3:
            confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        
        # Display predictions if available
        predictions_key = f"predictions_{st.session_state.current_symbol}_{st.session_state.selected_model}"
        
        if predictions_key in st.session_state.data_cache:
            predictions = st.session_state.data_cache[predictions_key][1]
            self._display_predictions(predictions)
        else:
            st.info("Click 'GENERATE PREDICTIONS' to run AI models on current symbol")
            
            # Show model capabilities
            self._show_model_capabilities()
    
    def _generate_predictions(self):
        """Generate AI predictions using the API."""
        try:
            with st.spinner(f"Training {st.session_state.selected_model} model..."):
                response = requests.post(
                    f"{self.api_base_url}/api/v1/predictions",
                    json={
                        "symbol": st.session_state.current_symbol,
                        "model_type": st.session_state.selected_model,
                        "horizon_days": st.session_state.prediction_horizon,
                        "confidence_interval": 0.95
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    predictions = response.json()
                    cache_key = f"predictions_{st.session_state.current_symbol}_{st.session_state.selected_model}"
                    st.session_state.data_cache[cache_key] = (datetime.now(), predictions)
                    st.success("✅ Predictions generated successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Prediction API error: {response.status_code}")
                    
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
    
    def _display_predictions(self, predictions):
        """Display prediction results with interactive charts."""
        # Prediction metrics
        col1, col2, col3, col4 = st.columns(4)
        
        model_confidence = predictions.get('model_confidence', 0)
        pred_data = predictions.get('predictions', [])
        
        if pred_data:
            next_day = pred_data[0]['predicted_price']
            week_pred = pred_data[6]['predicted_price'] if len(pred_data) > 6 else next_day
            final_pred = pred_data[-1]['predicted_price']
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MODEL CONFIDENCE</div>
                    <div class="metric-value">{model_confidence:.1%}</div>
                    <div class="metric-change positive">AI Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">NEXT DAY</div>
                    <div class="metric-value">${next_day:.2f}</div>
                    <div class="metric-change neutral">T+1 Forecast</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">1 WEEK</div>
                    <div class="metric-value">${week_pred:.2f}</div>
                    <div class="metric-change neutral">T+7 Forecast</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                horizon = predictions.get('horizon_days', 30)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{horizon}D TARGET</div>
                    <div class="metric-value">${final_pred:.2f}</div>
                    <div class="metric-change neutral">Long-term</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Prediction chart
        self._render_prediction_chart(predictions)
        
        # Model diagnostics
        self._render_model_diagnostics(predictions)
    
    def _render_prediction_chart(self, predictions):
        """Render interactive prediction chart."""
        pred_data = predictions.get('predictions', [])
        if not pred_data:
            return
        
        dates = [p['date'] for p in pred_data]
        prices = [p['predicted_price'] for p in pred_data]
        upper_bounds = [p['confidence_upper'] for p in pred_data]
        lower_bounds = [p['confidence_lower'] for p in pred_data]
        
        fig = go.Figure()
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            hoverinfo="skip"
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"{st.session_state.current_symbol} - {predictions.get('model_type', '').upper()} Predictions",
            template='plotly_dark',
            paper_bgcolor='rgba(42, 49, 66, 0.8)',
            plot_bgcolor='rgba(30, 35, 48, 0.8)',
            font=dict(color='#e8eaed'),
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_diagnostics(self, predictions):
        """Render model performance diagnostics."""
        st.markdown("### 🔬 MODEL DIAGNOSTICS")
        
        training_metrics = predictions.get('training_metrics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Performance**")
            metrics_df = pd.DataFrame({
                'Metric': ['Train Loss', 'Test Loss', 'R² Score'],
                'Value': [
                    training_metrics.get('final_train_loss', 0),
                    training_metrics.get('final_test_loss', 0),
                    training_metrics.get('r2_score', 0)
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            st.markdown("**Model Information**")
            info_df = pd.DataFrame({
                'Property': ['Model Type', 'Horizon', 'Generated At'],
                'Value': [
                    predictions.get('model_type', '').upper(),
                    f"{predictions.get('horizon_days', 0)} days",
                    predictions.get('generated_at', '')[:19]
                ]
            })
            st.dataframe(info_df, hide_index=True)
    
    def _show_model_capabilities(self):
        """Show available AI model capabilities."""
        st.markdown("### 🧠 AVAILABLE AI MODELS")
        
        models_info = {
            'LSTM': {
                'description': 'Long Short-Term Memory networks for sequential pattern learning',
                'strengths': 'Time series patterns, trend analysis',
                'use_case': 'Medium-term predictions (1-30 days)'
            },
            'Transformer': {
                'description': 'Attention-based architecture for complex pattern recognition',
                'strengths': 'Multi-variate analysis, market regime detection',
                'use_case': 'Complex market predictions'
            },
            'XGBoost': {
                'description': 'Gradient boosting for feature-rich predictions',
                'strengths': 'Feature importance, fast training',
                'use_case': 'Factor-based predictions'
            },
            'Ensemble': {
                'description': 'Combined predictions from multiple models',
                'strengths': 'Robust predictions, reduced overfitting',
                'use_case': 'Production forecasting'
            }
        }
        
        for model, info in models_info.items():
            st.markdown(f"""
            **{model}**: {info['description']}
            - *Strengths*: {info['strengths']}
            - *Best for*: {info['use_case']}
            """)
    
    def _render_portfolio_tab(self):
        """Portfolio optimization tab."""
        st.markdown("## 📊 PORTFOLIO OPTIMIZATION & ANALYSIS")
        # Implementation continues...
        
    def _render_valuation_tab(self):
        """DCF valuation tab."""
        st.markdown("## 💰 DCF VALUATION & FUNDAMENTAL ANALYSIS")
        # Implementation continues...
    
    def _render_risk_analysis_tab(self):
        """Risk analysis tab."""
        st.markdown("## ⚠️ ADVANCED RISK MANAGEMENT")
        # Implementation continues...
    
    def _render_backtesting_tab(self):
        """Backtesting tab."""
        st.markdown("## 🔄 STRATEGY BACKTESTING ENGINE")
        # Implementation continues...
    
    def _render_rl_agents_tab(self):
        """RL agents tab."""
        st.markdown("## 🎮 REINFORCEMENT LEARNING AGENTS")
        # Implementation continues...
    
    def _render_news_nlp_tab(self):
        """News and NLP tab."""
        st.markdown("## 📰 NEWS SENTIMENT & NLP ANALYSIS")
        # Implementation continues...
    
    def _render_reports_tab(self):
        """Reports tab."""
        st.markdown("## 📋 AUTOMATED REPORTING SYSTEM")
        # Implementation continues...
    
    def _render_llm_assistant_tab(self):
        """LLM assistant tab."""
        st.markdown("## 🤖 AI TRADING ASSISTANT")
        # Implementation continues...
    
    def _render_market_grid(self, market_data):
        """Render market data grid."""
        st.markdown("### 📊 Market Summary")
        # Implementation continues...
    
    def _render_enhanced_watchlist(self):
        """Render enhanced watchlist."""
        st.markdown("### 👁️ Watchlist")
        # Implementation continues...
    
    def _clear_cache(self):
        """Clear data cache."""
        st.session_state.data_cache = {}
        st.session_state.last_refresh = datetime.now()
    
    def _handle_auto_refresh(self):
        """Handle auto-refresh functionality."""
        if st.session_state.auto_refresh:
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
            if time_since_refresh >= st.session_state.refresh_interval:
                self._clear_cache()
                st.experimental_rerun()

def main():
    """Main application entry point."""
    terminal = EliteTerminal()
    terminal.run()

if __name__ == "__main__":
    main()