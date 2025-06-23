"""
Session State Manager for Bloomberg-Style Terminal
Manages persistent state across UI interactions and real-time updates.
"""

import streamlit as st
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages Streamlit session state for terminal application."""
    
    # Default session values
    DEFAULTS = {
        # Current symbol and market data
        'current_symbol': 'AAPL',
        'watchlist': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META'],
        'last_update': None,
        
        # UI state
        'active_tab': 'Market Data',
        'sidebar_expanded': True,
        'auto_refresh': True,
        'refresh_interval': 30,  # seconds
        
        # Market data settings
        'timeframe': '1Y',
        'chart_type': 'candlestick',
        'show_volume': True,
        'technical_indicators': ['RSI', 'MACD'],
        
        # AI/ML settings
        'selected_model': 'ensemble',
        'prediction_horizon': 30,  # days
        'confidence_level': 0.95,
        'model_cache': {},
        
        # Portfolio settings
        'portfolio_symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'portfolio_weights': [0.33, 0.33, 0.34],
        'risk_tolerance': 'moderate',
        'optimization_method': 'mean_variance',
        'rebalance_frequency': 'monthly',
        
        # Risk management settings
        'var_method': 'historical',
        'var_confidence': 0.95,
        'stress_test_scenarios': ['market_crash', 'recession', 'volatility_spike'],
        'position_size_method': 'kelly',
        'max_position_size': 0.1,
        
        # Backtesting settings
        'backtest_strategy': 'momentum',
        'backtest_start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        'backtest_end_date': datetime.now().strftime('%Y-%m-%d'),
        'initial_capital': 100000,
        
        # RL settings
        'rl_agent_type': 'TD3',
        'training_episodes': 100,
        'rl_environment': 'trading_v1',
        'agent_cache': {},
        
        # News and NLP settings
        'news_sources': ['reuters', 'bloomberg', 'marketwatch'],
        'sentiment_model': 'finbert',
        'news_lookback_days': 7,
        'sentiment_threshold': 0.1,
        
        # Reporting settings
        'report_type': 'market_summary',
        'report_period': '1M',
        'auto_generate_reports': False,
        'export_format': 'PDF',
        
        # LLM Assistant settings
        'llm_model': 'gpt-3.5-turbo',
        'conversation_history': [],
        'assistant_personality': 'professional',
        'max_conversation_length': 50,
        
        # Data cache
        'market_data_cache': {},
        'predictions_cache': {},
        'portfolio_cache': {},
        'risk_cache': {},
        'news_cache': {},
        
        # Performance tracking
        'page_load_time': None,
        'last_api_call': None,
        'error_count': 0,
        'warning_count': 0,
    }
    
    @classmethod
    def initialize(cls):
        """Initialize session state with default values."""
        for key, default_value in cls.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(f"Initialized session state: {key} = {default_value}")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get value from session state."""
        if key in st.session_state:
            return st.session_state[key]
        elif key in cls.DEFAULTS:
            return cls.DEFAULTS[key]
        else:
            return default
    
    @classmethod
    def set(cls, key: str, value: Any):
        """Set value in session state."""
        st.session_state[key] = value
        logger.debug(f"Updated session state: {key} = {value}")
    
    @classmethod
    def update(cls, updates: Dict[str, Any]):
        """Update multiple session state values."""
        for key, value in updates.items():
            cls.set(key, value)
    
    @classmethod
    def clear_cache(cls, cache_type: str = 'all'):
        """Clear cached data."""
        if cache_type == 'all':
            cache_keys = [k for k in st.session_state.keys() if k.endswith('_cache')]
            for key in cache_keys:
                st.session_state[key] = {}
        elif f'{cache_type}_cache' in st.session_state:
            st.session_state[f'{cache_type}_cache'] = {}
        
        logger.info(f"Cleared {cache_type} cache")
    
    @classmethod
    def get_cache(cls, cache_type: str) -> Dict:
        """Get cached data of specific type."""
        cache_key = f'{cache_type}_cache'
        return cls.get(cache_key, {})
    
    @classmethod
    def set_cache(cls, cache_type: str, key: str, value: Any, 
                  ttl_seconds: int = 300):
        """Set cached data with TTL."""
        cache_key = f'{cache_type}_cache'
        cache = cls.get(cache_key, {})
        
        cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl_seconds
        }
        
        cls.set(cache_key, cache)
    
    @classmethod
    def get_cached_value(cls, cache_type: str, key: str) -> Optional[Any]:
        """Get cached value if still valid."""
        cache = cls.get_cache(cache_type)
        
        if key in cache:
            entry = cache[key]
            if cls._is_cache_valid(entry):
                return entry['value']
            else:
                # Remove expired entry
                del cache[key]
                cls.set(f'{cache_type}_cache', cache)
        
        return None
    
    @classmethod
    def _is_cache_valid(cls, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        timestamp = cache_entry.get('timestamp')
        ttl = cache_entry.get('ttl', 300)
        
        if timestamp:
            expiry_time = timestamp + timedelta(seconds=ttl)
            return datetime.now() < expiry_time
        
        return False
    
    @classmethod
    def add_to_watchlist(cls, symbol: str):
        """Add symbol to watchlist."""
        watchlist = cls.get('watchlist', [])
        if symbol.upper() not in watchlist:
            watchlist.append(symbol.upper())
            cls.set('watchlist', watchlist)
            logger.info(f"Added {symbol} to watchlist")
    
    @classmethod
    def remove_from_watchlist(cls, symbol: str):
        """Remove symbol from watchlist."""
        watchlist = cls.get('watchlist', [])
        if symbol.upper() in watchlist:
            watchlist.remove(symbol.upper())
            cls.set('watchlist', watchlist)
            logger.info(f"Removed {symbol} from watchlist")
    
    @classmethod
    def update_portfolio(cls, symbols: List[str], weights: List[float]):
        """Update portfolio composition."""
        if len(symbols) == len(weights) and abs(sum(weights) - 1.0) < 0.01:
            cls.set('portfolio_symbols', symbols)
            cls.set('portfolio_weights', weights)
            logger.info(f"Updated portfolio: {dict(zip(symbols, weights))}")
        else:
            logger.error("Portfolio update failed: symbols and weights mismatch")
    
    @classmethod
    def log_error(cls, error_message: str):
        """Log error and increment error count."""
        error_count = cls.get('error_count', 0)
        cls.set('error_count', error_count + 1)
        logger.error(error_message)
    
    @classmethod
    def log_warning(cls, warning_message: str):
        """Log warning and increment warning count."""
        warning_count = cls.get('warning_count', 0)
        cls.set('warning_count', warning_count + 1)
        logger.warning(warning_message)
    
    @classmethod
    def reset_session(cls):
        """Reset session state to defaults."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        cls.initialize()
        logger.info("Session state reset to defaults")
    
    @classmethod
    def export_session(cls) -> Dict:
        """Export current session state for debugging."""
        return {
            key: value for key, value in st.session_state.items()
            if not key.endswith('_cache')  # Exclude large cache objects
        }
    
    @classmethod
    def get_session_info(cls) -> Dict:
        """Get session information summary."""
        return {
            'current_symbol': cls.get('current_symbol'),
            'active_tab': cls.get('active_tab'),
            'last_update': cls.get('last_update'),
            'error_count': cls.get('error_count'),
            'warning_count': cls.get('warning_count'),
            'cache_sizes': {
                cache_type: len(cls.get_cache(cache_type.replace('_cache', '')))
                for cache_type in st.session_state.keys()
                if cache_type.endswith('_cache')
            }
        }