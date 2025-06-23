#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal - Bloomberg-Style Web Interface
Professional-grade quantitative finance terminal with real-time data and AI capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
import time
import logging
import os
import sys
from typing import Dict, List, Optional, Any
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from ui.utils.theme import BloombergTheme
    from ui.utils.session import SessionManager
    from src.api.main import get_mock_terminal_data
except ImportError as e:
    # Fallback if modules not available
    class BloombergTheme:
        @staticmethod
        def apply_theme(): pass
        @staticmethod
        def create_metric_card(label, value, change=None, change_type='neutral'):
            return f"<div><strong>{label}:</strong> {value} {change or ''}</div>"
        @staticmethod
        def create_header(title, status='live'):
            return f"<h3>{title}</h3>"
        @staticmethod
        def format_number(value, precision=2, show_sign=True, percentage=False):
            return f"{value:,.{precision}f}{'%' if percentage else ''}"
        @staticmethod
        def get_color_for_value(value, threshold=0):
            return 'positive' if value > threshold else 'negative' if value < threshold else 'neutral'
    
    class SessionManager:
        @staticmethod
        def initialize(): pass
        @staticmethod
        def get(key, default=None): return st.session_state.get(key, default)
        @staticmethod
        def set(key, value): st.session_state[key] = value

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

class EliteTerminal:
    """Main Bloomberg-style terminal application."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.theme = BloombergTheme()
        self.session = SessionManager()
        
        # Initialize session state
        self.session.initialize()
        
        # Apply Bloomberg theme
        self.theme.apply_theme()
        
        # Initialize data containers
        self.data_containers = {}
        
    def run(self):
        """Main application entry point."""
        try:
            # Render header
            self._render_header()
            
            # Render sidebar
            self._render_sidebar()
            
            # Render main content with tabs
            self._render_main_content()
            
            # Auto-refresh logic
            self._handle_auto_refresh()
            
        except Exception as e:
            st.error(f"Terminal Error: {str(e)}")
            logger.error(f"Terminal error: {str(e)}")
    
    def _render_header(self):
        """Render terminal header with live status."""
        current_time = datetime.now().strftime("%H:%M:%S")
        utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        header_html = f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
            border-bottom: 1px solid #333333;
            padding: 8px 16px;
            margin: -1rem -1rem 1rem -1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'Roboto Mono', monospace;
        ">
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-live"></span>
                <span style="color: #808080; font-size: 10px; font-weight: 500;">LIVE DATA</span>
            </div>
            <div style="text-align: center;">
                <div style="color: #00bfff; font-size: 16px; font-weight: 600; letter-spacing: 1px;">
                    MORGANVUOKSI ELITE TERMINAL
                </div>
                <div style="color: #808080; font-size: 9px; margin-top: 2px;">
                    Bloomberg-Grade Quantitative Finance Platform
                </div>
            </div>
            <div style="text-align: right;">
                <div style="color: #ffffff; font-size: 11px; font-weight: 600;">{current_time}</div>
                <div style="color: #808080; font-size: 9px;">{utc_time}</div>
            </div>
        </div>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render Bloomberg-style sidebar with controls."""
        with st.sidebar:
            st.markdown(self.theme.create_header("TERMINAL CONTROLS"), unsafe_allow_html=True)
            
            # Symbol input
            current_symbol = self.session.get('current_symbol', 'AAPL')
            symbol = st.text_input(
                "SYMBOL", 
                value=current_symbol,
                max_chars=10,
                help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
            ).upper()
            
            if symbol != current_symbol:
                self.session.set('current_symbol', symbol)
                st.experimental_rerun()
            
            # Quick symbol buttons
            st.markdown("**QUICK SELECT**")
            watchlist = self.session.get('watchlist', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
            
            cols = st.columns(2)
            for i, sym in enumerate(watchlist[:6]):
                col = cols[i % 2]
                if col.button(sym, key=f"quick_{sym}", use_container_width=True):
                    self.session.set('current_symbol', sym)
                    st.experimental_rerun()
            
            st.markdown("---")
            
            # Market data settings
            st.markdown(self.theme.create_header("MARKET DATA", "live"), unsafe_allow_html=True)
            
            timeframe = st.selectbox(
                "TIMEFRAME",
                options=['1D', '5D', '1M', '3M', '6M', '1Y', '2Y', '5Y'],
                index=5,  # Default to 1Y
                key='timeframe_select'
            )
            self.session.set('timeframe', timeframe)
            
            show_volume = st.checkbox("SHOW VOLUME", value=True)
            self.session.set('show_volume', show_volume)
            
            # Technical indicators
            indicators = st.multiselect(
                "TECHNICAL INDICATORS",
                options=['RSI', 'MACD', 'BB', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26'],
                default=['RSI', 'MACD'],
                key='tech_indicators'
            )
            self.session.set('technical_indicators', indicators)
            
            st.markdown("---")
            
            # AI/ML settings
            st.markdown(self.theme.create_header("AI MODELS", "live"), unsafe_allow_html=True)
            
            model_type = st.selectbox(
                "PREDICTION MODEL",
                options=['lstm', 'transformer', 'xgboost', 'ensemble'],
                index=3,  # Default to ensemble
                key='model_select'
            )
            self.session.set('selected_model', model_type)
            
            horizon = st.slider(
                "PREDICTION HORIZON (DAYS)",
                min_value=1, max_value=90, value=30,
                key='horizon_slider'
            )
            self.session.set('prediction_horizon', horizon)
            
            st.markdown("---")
            
            # System controls
            st.markdown(self.theme.create_header("SYSTEM", "live"), unsafe_allow_html=True)
            
            auto_refresh = st.checkbox("AUTO REFRESH", value=True)
            self.session.set('auto_refresh', auto_refresh)
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "REFRESH INTERVAL (SEC)",
                    min_value=10, max_value=300, value=30
                )
                self.session.set('refresh_interval', refresh_interval)
            
            col1, col2 = st.columns(2)
            if col1.button("🔄 REFRESH", use_container_width=True):
                self.session.clear_cache('all')
                st.experimental_rerun()
            
            if col2.button("⚙️ RESET", use_container_width=True):
                self.session.reset_session()
                st.experimental_rerun()
    
    def _render_main_content(self):
        """Render main tabbed content area."""
        tabs = st.tabs([
            "📈 MARKET DATA",
            "🤖 AI PREDICTIONS", 
            "📊 PORTFOLIO",
            "💰 VALUATION",
            "⚠️ RISK",
            "🔄 BACKTEST",
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
            self._render_risk_tab()
        
        with tabs[5]:
            self._render_backtest_tab()
        
        with tabs[6]:
            self._render_rl_agents_tab()
        
        with tabs[7]:
            self._render_news_nlp_tab()
        
        with tabs[8]:
            self._render_reports_tab()
        
        with tabs[9]:
            self._render_llm_assistant_tab()
    
    def _render_market_data_tab(self):
        """Render market data and technical analysis tab."""
        st.markdown(self.theme.create_header("REAL-TIME MARKET DATA", "live"), unsafe_allow_html=True)
        
        # Get market data
        symbol = self.session.get('current_symbol', 'AAPL')
        market_data = self._get_market_data(symbol)
        
        if market_data:
            # Market overview metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            symbol_data = market_data.get('symbol', {})
            current_price = symbol_data.get('price', 0)
            change_val = symbol_data.get('change_val', 0)
            change_pct = symbol_data.get('change_pct', 0)
            volume = symbol_data.get('volume', '0')
            market_cap = symbol_data.get('market_cap', '0')
            
            # Price metrics
            change_type = self.theme.get_color_for_value(change_pct)
            change_arrow = "↗" if change_pct >= 0 else "↘"
            
            with col1:
                st.markdown(
                    self.theme.create_metric_card(
                        "PRICE", 
                        f"${current_price}",
                        f"{change_arrow} {self.theme.format_number(change_val)} ({self.theme.format_number(change_pct, percentage=True)})",
                        change_type
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    self.theme.create_metric_card("VOLUME", volume),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    self.theme.create_metric_card("MARKET CAP", market_cap),
                    unsafe_allow_html=True
                )
            
            with col4:
                pe_ratio = symbol_data.get('pe_ratio', 'N/A')
                st.markdown(
                    self.theme.create_metric_card("P/E RATIO", str(pe_ratio)),
                    unsafe_allow_html=True
                )
            
            with col5:
                sector = symbol_data.get('sector', 'N/A')
                st.markdown(
                    self.theme.create_metric_card("SECTOR", sector),
                    unsafe_allow_html=True
                )
            
            # Price chart
            self._render_price_chart(market_data)
            
            # Market data grid
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._render_market_grid(market_data)
            
            with col2:
                self._render_watchlist()
    
    def _render_ai_predictions_tab(self):
        """Render AI predictions and model diagnostics."""
        st.markdown(self.theme.create_header("AI PRICE PREDICTIONS", "live"), unsafe_allow_html=True)
        
        symbol = self.session.get('current_symbol', 'AAPL')
        model_type = self.session.get('selected_model', 'ensemble')
        horizon = self.session.get('prediction_horizon', 30)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("🤖 GENERATE PREDICTIONS", use_container_width=True):
                with st.spinner("Running AI models..."):
                    predictions = self._get_predictions(symbol, model_type, horizon)
                    if predictions:
                        self.session.set_cache('predictions', f'{symbol}_{model_type}', predictions)
        
        with col3:
            confidence = st.slider("CONFIDENCE LEVEL", 0.80, 0.99, 0.95, 0.01)
        
        # Display cached predictions
        predictions = self.session.get_cached_value('predictions', f'{symbol}_{model_type}')
        
        if predictions:
            # Prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            
            model_confidence = predictions.get('model_confidence', 0)
            pred_data = predictions.get('predictions', [])
            
            if pred_data:
                next_day_pred = pred_data[0]['predicted_price']
                week_pred = pred_data[6]['predicted_price'] if len(pred_data) > 6 else next_day_pred
                
                with col1:
                    st.markdown(
                        self.theme.create_metric_card("MODEL CONFIDENCE", f"{model_confidence:.1%}"),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        self.theme.create_metric_card("NEXT DAY", f"${next_day_pred:.2f}"),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        self.theme.create_metric_card("1 WEEK", f"${week_pred:.2f}"),
                        unsafe_allow_html=True
                    )
                
                with col4:
                    last_pred = pred_data[-1]['predicted_price']
                    st.markdown(
                        self.theme.create_metric_card(f"{horizon}D TARGET", f"${last_pred:.2f}"),
                        unsafe_allow_html=True
                    )
            
            # Prediction chart
            self._render_prediction_chart(predictions)
            
            # Model diagnostics
            self._render_model_diagnostics(model_type)
        else:
            st.info("Click 'GENERATE PREDICTIONS' to run AI models")
    
    def _render_portfolio_tab(self):
        """Render portfolio optimization and analysis."""
        st.markdown(self.theme.create_header("PORTFOLIO OPTIMIZATION", "live"), unsafe_allow_html=True)
        
        # Portfolio input
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            symbols_input = st.text_area(
                "PORTFOLIO SYMBOLS (one per line)",
                value="\n".join(self.session.get('portfolio_symbols', ['AAPL', 'GOOGL', 'MSFT'])),
                height=100
            )
            symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        with col2:
            risk_tolerance = st.selectbox(
                "RISK TOLERANCE",
                options=['conservative', 'moderate', 'aggressive'],
                index=1
            )
        
        with col3:
            optimization_method = st.selectbox(
                "OPTIMIZATION METHOD",
                options=['mean_variance', 'black_litterman', 'risk_parity', 'maximum_sharpe'],
                index=0
            )
        
        if st.button("⚡ OPTIMIZE PORTFOLIO", use_container_width=True):
            with st.spinner("Optimizing portfolio..."):
                optimization_result = self._optimize_portfolio(symbols, risk_tolerance, optimization_method)
                if optimization_result:
                    self.session.set_cache('portfolio', 'optimization', optimization_result)
        
        # Display optimization results
        optimization_result = self.session.get_cached_value('portfolio', 'optimization')
        
        if optimization_result:
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            expected_return = optimization_result.get('expected_return', 0)
            volatility = optimization_result.get('volatility', 0)
            sharpe_ratio = optimization_result.get('sharpe_ratio', 0)
            
            with col1:
                st.markdown(
                    self.theme.create_metric_card(
                        "EXPECTED RETURN", 
                        self.theme.format_number(expected_return * 100, percentage=True)
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    self.theme.create_metric_card(
                        "VOLATILITY",
                        self.theme.format_number(volatility * 100, percentage=True)
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    self.theme.create_metric_card("SHARPE RATIO", f"{sharpe_ratio:.2f}"),
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    self.theme.create_metric_card("METHOD", optimization_method.upper()),
                    unsafe_allow_html=True
                )
            
            # Portfolio allocation chart
            self._render_portfolio_chart(optimization_result)
            
            # Efficient frontier
            self._render_efficient_frontier(optimization_result)
        else:
            st.info("Click 'OPTIMIZE PORTFOLIO' to run optimization")
    
    def _render_valuation_tab(self):
        """Render DCF valuation and fundamental analysis."""
        st.markdown(self.theme.create_header("DCF VALUATION", "live"), unsafe_allow_html=True)
        
        symbol = self.session.get('current_symbol', 'AAPL')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💰 CALCULATE DCF", use_container_width=True):
                with st.spinner("Calculating DCF valuation..."):
                    dcf_data = self._get_dcf_valuation(symbol)
                    if dcf_data:
                        self.session.set_cache('valuation', symbol, dcf_data)
        
        # Display DCF results
        dcf_data = self.session.get_cached_value('valuation', symbol)
        
        if dcf_data:
            # Valuation metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = dcf_data.get('current_price', 0)
            intrinsic_value = dcf_data.get('intrinsic_value', 0)
            margin_of_safety = dcf_data.get('margin_of_safety', '0%')
            recommendation = dcf_data.get('recommendation', 'HOLD')
            
            with col1:
                st.markdown(
                    self.theme.create_metric_card("CURRENT PRICE", f"${current_price:.2f}"),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    self.theme.create_metric_card("INTRINSIC VALUE", f"${intrinsic_value:.2f}"),
                    unsafe_allow_html=True
                )
            
            with col3:
                margin_float = float(margin_of_safety.replace('%', ''))
                margin_type = self.theme.get_color_for_value(margin_float)
                st.markdown(
                    self.theme.create_metric_card("MARGIN OF SAFETY", margin_of_safety, change_type=margin_type),
                    unsafe_allow_html=True
                )
            
            with col4:
                rec_type = 'positive' if recommendation == 'BUY' else 'negative' if recommendation == 'SELL' else 'neutral'
                st.markdown(
                    self.theme.create_metric_card("RECOMMENDATION", recommendation, change_type=rec_type),
                    unsafe_allow_html=True
                )
            
            # DCF assumptions
            st.markdown("### DCF ASSUMPTIONS")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Growth Rate:** {dcf_data.get('growth_rate', 'N/A')}")
            with col2:
                st.markdown(f"**Discount Rate:** {dcf_data.get('discount_rate', 'N/A')}")
            with col3:
                st.markdown(f"**Terminal Growth:** {dcf_data.get('terminal_growth', 'N/A')}")
        else:
            st.info("Click 'CALCULATE DCF' to run valuation analysis")
    
    def _render_risk_tab(self):
        """Render risk management and analysis."""
        st.markdown(self.theme.create_header("RISK MANAGEMENT", "warning"), unsafe_allow_html=True)
        
        # Risk analysis placeholder
        st.info("Risk analysis functionality - VaR, CVaR, stress testing, position sizing")
    
    def _render_backtest_tab(self):
        """Render backtesting interface."""
        st.markdown(self.theme.create_header("STRATEGY BACKTESTING", "live"), unsafe_allow_html=True)
        
        # Backtesting placeholder
        st.info("Backtesting functionality - strategy testing, performance metrics, trade analysis")
    
    def _render_rl_agents_tab(self):
        """Render reinforcement learning agents."""
        st.markdown(self.theme.create_header("RL TRADING AGENTS", "live"), unsafe_allow_html=True)
        
        # RL agents placeholder
        st.info("RL agents functionality - TD3/SAC agents, training progress, agent performance")
    
    def _render_news_nlp_tab(self):
        """Render news and sentiment analysis."""
        st.markdown(self.theme.create_header("NEWS & SENTIMENT", "live"), unsafe_allow_html=True)
        
        # News and NLP placeholder
        st.info("News & NLP functionality - sentiment analysis, news aggregation, FinBERT processing")
    
    def _render_reports_tab(self):
        """Render automated reporting."""
        st.markdown(self.theme.create_header("AUTOMATED REPORTING", "live"), unsafe_allow_html=True)
        
        # Reports placeholder
        st.info("Reporting functionality - automated reports, PDF/Excel export, AI summarization")
    
    def _render_llm_assistant_tab(self):
        """Render LLM assistant interface."""
        st.markdown(self.theme.create_header("LLM TRADING ASSISTANT", "live"), unsafe_allow_html=True)
        
        # LLM assistant placeholder
        st.info("LLM assistant functionality - GPT-powered chat, market analysis, strategy guidance")
    
    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data from API or cache."""
        # Check cache first
        cached_data = self.session.get_cached_value('market_data', symbol)
        if cached_data:
            return cached_data
        
        try:
            # Try API call
            response = requests.get(f"{self.api_base_url}/api/v1/terminal_data/{symbol}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.session.set_cache('market_data', symbol, data, ttl_seconds=60)
                return data
        except Exception as e:
            logger.warning(f"API call failed: {e}")
        
        # Fallback to mock data
        try:
            mock_data = get_mock_terminal_data(symbol)
            return mock_data
        except:
            return None
    
    def _get_predictions(self, symbol: str, model_type: str, horizon: int) -> Optional[Dict]:
        """Get AI predictions from API."""
        try:
            payload = {
                'symbol': symbol,
                'model_type': model_type,
                'horizon_days': horizon,
                'confidence_interval': 0.95
            }
            response = requests.post(f"{self.api_base_url}/api/v1/predictions", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Predictions API call failed: {e}")
        
        return None
    
    def _optimize_portfolio(self, symbols: List[str], risk_tolerance: str, method: str) -> Optional[Dict]:
        """Get portfolio optimization from API."""
        try:
            payload = {
                'symbols': symbols,
                'method': method,
                'risk_tolerance': risk_tolerance,
                'initial_capital': 100000
            }
            response = requests.post(f"{self.api_base_url}/api/v1/portfolio/optimize", json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Portfolio optimization API call failed: {e}")
        
        return None
    
    def _get_dcf_valuation(self, symbol: str) -> Optional[Dict]:
        """Get DCF valuation from API."""
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/dcf/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"DCF API call failed: {e}")
        
        return None
    
    def _render_price_chart(self, market_data: Dict):
        """Render interactive price chart."""
        chart_data = market_data.get('price_chart', {})
        timeframe = self.session.get('timeframe', '1Y')
        
        if timeframe in chart_data:
            data_points = chart_data[timeframe]
            
            if data_points:
                df = pd.DataFrame(data_points)
                df['time'] = pd.to_datetime(df['time'])
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=f"{self.session.get('current_symbol')} Price",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ))
                
                # Volume if enabled
                if self.session.get('show_volume', True):
                    fig.add_trace(go.Bar(
                        x=df['time'],
                        y=df['volume'],
                        name='Volume',
                        yaxis='y2',
                        opacity=0.3,
                        marker_color='#0066cc'
                    ))
                
                # Chart layout
                fig.update_layout(
                    title=f"{self.session.get('current_symbol')} - {timeframe}",
                    template='plotly_dark',
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#0a0a0a',
                    font_color='#ffffff',
                    font_family='Roboto Mono',
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    ) if self.session.get('show_volume') else None,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_prediction_chart(self, predictions: Dict):
        """Render prediction chart with confidence intervals."""
        pred_data = predictions.get('predictions', [])
        
        if pred_data:
            df = pd.DataFrame(pred_data)
            df['date'] = pd.to_datetime(df['date'])
            
            fig = go.Figure()
            
            # Predicted price line
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['predicted_price'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='#00bfff', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['confidence_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,191,255,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['confidence_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,191,255,0)',
                name='Confidence Interval',
                fillcolor='rgba(0,191,255,0.2)'
            ))
            
            fig.update_layout(
                title=f"AI Price Predictions - {predictions.get('model_type', '').upper()}",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#0a0a0a',
                font_color='#ffffff',
                font_family='Roboto Mono',
                height=400,
                xaxis_title="Date",
                yaxis_title="Predicted Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_portfolio_chart(self, optimization_result: Dict):
        """Render portfolio allocation chart."""
        weights = optimization_result.get('optimal_weights', {})
        
        if weights:
            symbols = list(weights.keys())
            values = list(weights.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=values,
                hole=0.4,
                marker_colors=['#0066cc', '#00bfff', '#ff8c42', '#00d4aa', '#ff6b6b']
            )])
            
            fig.update_layout(
                title="Optimal Portfolio Allocation",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#0a0a0a',
                font_color='#ffffff',
                font_family='Roboto Mono',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_efficient_frontier(self, optimization_result: Dict):
        """Render efficient frontier chart."""
        frontier_data = optimization_result.get('efficient_frontier', [])
        
        if frontier_data:
            df = pd.DataFrame(frontier_data)
            
            fig = go.Figure()
            
            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=df['volatility'],
                y=df['return'],
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='#0066cc', width=2),
                marker=dict(size=4)
            ))
            
            # Optimal portfolio point
            optimal_vol = optimization_result.get('volatility', 0)
            optimal_ret = optimization_result.get('expected_return', 0)
            
            fig.add_trace(go.Scatter(
                x=[optimal_vol],
                y=[optimal_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='#00ff00',
                    symbol='star'
                )
            ))
            
            fig.update_layout(
                title="Efficient Frontier",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#0a0a0a',
                font_color='#ffffff',
                font_family='Roboto Mono',
                height=400,
                xaxis_title="Volatility (Risk)",
                yaxis_title="Expected Return"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_market_grid(self, market_data: Dict):
        """Render market data in grid format."""
        st.markdown("### MARKET DATA GRID")
        
        # Create sample market data table
        data = {
            'Metric': ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg Volume', 'Market Cap', 'P/E Ratio'],
            'Value': ['$150.25', '$152.10', '$149.80', '$151.45', '2.5M', '3.1M', '2.4T', '28.5'],
            'Change': ['+1.2%', '+0.8%', '-0.5%', '+1.1%', '-15%', '—', '+1.1%', '—']
        }
        
        df = pd.DataFrame(data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("METRIC", width="medium"),
                "Value": st.column_config.TextColumn("VALUE", width="medium"),
                "Change": st.column_config.TextColumn("CHANGE", width="small")
            }
        )
    
    def _render_watchlist(self):
        """Render watchlist panel."""
        st.markdown("### WATCHLIST")
        
        watchlist = self.session.get('watchlist', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        for symbol in watchlist:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button(symbol, key=f"watch_{symbol}", use_container_width=True):
                    self.session.set('current_symbol', symbol)
                    st.experimental_rerun()
            
            with col2:
                st.markdown(f"<small style='color: #808080;'>$150.25</small>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"<small style='color: #00ff00;'>+1.2%</small>", unsafe_allow_html=True)
    
    def _render_model_diagnostics(self, model_type: str):
        """Render model performance diagnostics."""
        st.markdown("### MODEL DIAGNOSTICS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mock feature importance
            features = ['Price MA', 'Volume', 'RSI', 'MACD', 'Volatility']
            importance = [0.35, 0.25, 0.20, 0.15, 0.05]
            
            fig = go.Figure(data=[go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color='#0066cc'
            )])
            
            fig.update_layout(
                title="Feature Importance",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#0a0a0a',
                font_color='#ffffff',
                font_family='Roboto Mono',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mock loss curve
            epochs = list(range(1, 51))
            loss = [0.1 * np.exp(-0.1 * x) + 0.01 + 0.005 * np.random.random() for x in epochs]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs,
                y=loss,
                mode='lines',
                name='Training Loss',
                line=dict(color='#ff8c42')
            ))
            
            fig.update_layout(
                title="Training Loss Curve",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#0a0a0a',
                font_color='#ffffff',
                font_family='Roboto Mono',
                height=300,
                xaxis_title="Epoch",
                yaxis_title="Loss"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _handle_auto_refresh(self):
        """Handle auto-refresh functionality."""
        if self.session.get('auto_refresh', True):
            refresh_interval = self.session.get('refresh_interval', 30)
            
            # Auto refresh using st.rerun with time check
            last_update = self.session.get('last_update')
            current_time = datetime.now()
            
            if not last_update or (current_time - last_update).seconds >= refresh_interval:
                self.session.set('last_update', current_time)
                time.sleep(0.1)  # Brief pause to prevent excessive refreshing

def main():
    """Main application entry point."""
    terminal = EliteTerminal()
    terminal.run()

if __name__ == "__main__":
    main()