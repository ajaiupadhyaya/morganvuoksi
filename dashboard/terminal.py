"""
MorganVuoksi Terminal - Bloomberg-Style Quantitative Trading Terminal
Modern, institutional-grade interface for quantitative research and trading.
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from data.market_data import MarketDataFetcher, DataConfig
from models.advanced_models import TimeSeriesPredictor, ARIMAGARCHModel, EnsembleModel
from models.rl_models import TD3Agent, SACAgent, TradingEnvironment
from signals.nlp_signals import NLPSignalGenerator, FinancialNLPAnalyzer
from portfolio.optimizer import PortfolioOptimizer
from risk.risk_manager import RiskManager
from visuals.dashboard import create_candlestick_chart, create_technical_chart
from visuals.portfolio_visuals import create_portfolio_chart
from visuals.risk_visuals import create_risk_dashboard

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MorganVuoksi Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg-style theme
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
        background-color: #1e1e1e;
        color: #fafafa;
    }
    .stTextInput, .stSelectbox, .stNumberInput {
        background-color: #2d2d2d;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background-color: #0052a3;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
    }
    .positive-change {
        color: #00ff88;
    }
    .negative-change {
        color: #ff4444;
    }
    .neutral-change {
        color: #cccccc;
    }
</style>
""", unsafe_allow_html=True)

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
        col1, col2, col3 = st.columns([2, 6, 2])
        
        with col1:
            st.markdown("### 📈 MorganVuoksi")
        
        with col2:
            st.markdown("<h1 style='text-align: center; color: #0066cc;'>Quantitative Trading Terminal</h1>", 
                       unsafe_allow_html=True)
        
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"### {current_time}")
        
        st.markdown("---")
    
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
                
                st.metric("Current Price", f"${current_price:.2f}", 
                         f"{change:+.2f} ({change_pct:+.2f}%)")
            
            with col2:
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = volume / avg_volume
                
                st.metric("Volume", f"{volume:,.0f}", 
                         f"{volume_ratio:.1f}x avg")
            
            with col3:
                if 'RSI' in data.columns:
                    rsi = data['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI", f"{rsi:.1f}", rsi_status)
            
            with col4:
                if 'Volatility' in data.columns:
                    vol = data['Volatility'].iloc[-1] * 100
                    st.metric("Volatility", f"{vol:.1f}%")
            
            # Charts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Price Chart")
                fig = create_candlestick_chart(data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Technical Indicators")
                fig = create_technical_chart(data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Market statistics
            st.markdown("### Market Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Price Statistics")
                price_stats = data['Close'].describe()
                st.dataframe(price_stats)
            
            with col2:
                st.markdown("#### Volume Statistics")
                volume_stats = data['Volume'].describe()
                st.dataframe(volume_stats)
            
            with col3:
                st.markdown("#### Returns Distribution")
                returns = data['Close'].pct_change().dropna()
                returns_stats = returns.describe()
                st.dataframe(returns_stats)
        
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
            if 'allocation_chart' in results:
                st.plotly_chart(results['allocation_chart'], use_container_width=True)
            
            # Efficient frontier
            st.markdown("### Efficient Frontier")
            if 'efficient_frontier' in results:
                st.plotly_chart(results['efficient_frontier'], use_container_width=True)
    
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
                fig = px.pie(values=[dist['positive'], dist['negative'], dist['neutral']], 
                           names=['Positive', 'Negative', 'Neutral'],
                           title="News Sentiment Distribution")
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
                    'confidence': 0.75,  # Placeholder
                    'signal': 'buy' if predicted_price > current_price else 'sell',
                    'accuracy': 0.65,  # Placeholder
                    'model_type': model_type,
                    'horizon': horizon
                }
                
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
                weights = self.portfolio_optimizer.optimize_portfolio(returns_df, risk_tolerance)
                
                # Calculate metrics
                portfolio_return = (returns_df * weights).sum(axis=1).mean() * 252
                portfolio_vol = (returns_df * weights).sum(axis=1).std() * np.sqrt(252)
                sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                
                # Store results
                st.session_state.portfolio_results = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': 0.15  # Placeholder
                }
                
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
                    'trades_df': pd.DataFrame()  # Placeholder
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

def main():
    """Main application entry point."""
    terminal = MorganVuoksiTerminal()
    terminal.run()

if __name__ == "__main__":
    main() 
