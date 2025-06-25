#!/usr/bin/env python3
"""
MorganVuoksi Performance Optimization & ML/AI Supercharger
Optimizes the terminal for web hosting and enhances ML/AI capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import functools
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
import pickle
import hashlib
import os
from pathlib import Path
import gc
import psutil

# Configure logging for performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Advanced performance optimization for web deployment."""
    
    def __init__(self):
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_threshold = 85  # Memory usage threshold %
        self.cache_ttl = 300  # 5 minutes default TTL
        
    def enhanced_cache(self, ttl: int = 300, key_func: Optional[Callable] = None):
        """Enhanced caching decorator with memory management."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                # Check if cached result exists and is valid
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        if time.time() - cached_data['timestamp'] < ttl:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return cached_data['result']
                    except Exception as e:
                        logger.warning(f"Cache read error: {e}")
                
                # Check memory usage before execution
                self._check_memory_usage()
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache result
                try:
                    cached_data = {
                        'result': result,
                        'timestamp': time.time(),
                        'execution_time': execution_time
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cached_data, f)
                    logger.debug(f"Cached result for {func.__name__} (exec: {execution_time:.2f}s)")
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key for function call."""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_memory_usage(self):
        """Monitor and manage memory usage."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.memory_threshold:
            logger.warning(f"High memory usage: {memory_percent}%")
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Cleanup memory and cache."""
        # Force garbage collection
        gc.collect()
        
        # Clear old cache files
        current_time = time.time()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if current_time - cache_file.stat().st_mtime > self.cache_ttl * 2:
                    cache_file.unlink()
                    logger.debug(f"Removed old cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Error removing cache file: {e}")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df.empty:
            return df
            
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df

class MLAISupercharger:
    """Advanced ML/AI capabilities for the terminal."""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced financial features using AI techniques."""
        features = data.copy()
        
        # Technical indicators with AI enhancement
        features = self._add_technical_features(features)
        
        # Microstructure features
        features = self._add_microstructure_features(features)
        
        # Sentiment features (if available)
        features = self._add_sentiment_features(features)
        
        # Regime detection features
        features = self._add_regime_features(features)
        
        # Cross-asset features
        features = self._add_cross_asset_features(features)
        
        return features
    
    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical analysis features."""
        if 'Close' not in data.columns:
            return data
        
        # Advanced moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{window}'] = data['Close'].rolling(window).mean()
            data[f'ema_{window}'] = data['Close'].ewm(span=window).mean()
        
        # Bollinger Bands with multiple deviations
        for std_dev in [1, 2, 3]:
            sma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            data[f'bb_upper_{std_dev}'] = sma_20 + (std_dev * std_20)
            data[f'bb_lower_{std_dev}'] = sma_20 - (std_dev * std_20)
            data[f'bb_position_{std_dev}'] = (data['Close'] - data[f'bb_lower_{std_dev}']) / (data[f'bb_upper_{std_dev}'] - data[f'bb_lower_{std_dev}'])
        
        # RSI with multiple periods
        for period in [14, 21, 30]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_fast = data['Close'].ewm(span=fast).mean()
            ema_slow = data['Close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            data[f'macd_{fast}_{slow}'] = macd_line
            data[f'macd_signal_{fast}_{slow}'] = signal_line
            data[f'macd_histogram_{fast}_{slow}'] = macd_line - signal_line
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        if 'Volume' not in data.columns:
            return data
        
        # Volume-weighted average price
        if all(col in data.columns for col in ['High', 'Low', 'Close', 'Volume']):
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            data['vwap'] = (typical_price * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        # Volume features
        data['volume_sma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
        data['volume_momentum'] = data['Volume'].pct_change(5)
        
        # Price-volume features
        data['price_volume_correlation'] = data['Close'].rolling(20).corr(data['Volume'])
        
        return data
    
    def _add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features."""
        # Simulated sentiment features (in production, integrate with news APIs)
        sentiment_periods = [5, 10, 20]
        
        for period in sentiment_periods:
            # Volatility-based sentiment proxy
            data[f'volatility_sentiment_{period}'] = data['Close'].rolling(period).std() / data['Close'].rolling(period).mean()
            
            # Price momentum sentiment
            data[f'momentum_sentiment_{period}'] = data['Close'].pct_change(period)
        
        return data
    
    def _add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        if 'Close' not in data.columns:
            return data
        
        # Volatility regimes
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        vol_quantiles = rolling_vol.rolling(252).quantile([0.25, 0.75])
        
        data['vol_regime_low'] = (rolling_vol <= vol_quantiles.iloc[:, 0]).astype(int)
        data['vol_regime_high'] = (rolling_vol >= vol_quantiles.iloc[:, 1]).astype(int)
        
        # Trend regimes
        sma_short = data['Close'].rolling(20).mean()
        sma_long = data['Close'].rolling(50).mean()
        data['trend_regime'] = (sma_short > sma_long).astype(int)
        
        return data
    
    def _add_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation features."""
        # This would be enhanced with actual cross-asset data in production
        # For now, create proxy features
        
        if 'Close' not in data.columns:
            return data
        
        # Simulated cross-asset correlations
        returns = data['Close'].pct_change()
        
        # Rolling correlations with major indices (simulated)
        for window in [20, 60, 120]:
            data[f'cross_correlation_{window}'] = returns.rolling(window).corr(returns.shift(1))
        
        return data
    
    def create_ensemble_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Create an ensemble of ML models for predictions."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split, cross_val_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'svr': SVR(kernel='rbf')
        }
        
        # Train and evaluate models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': model.score(X_test, y_test)
                }
                
                logger.info(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Calculate ensemble weights based on performance
        ensemble_weights = self._calculate_ensemble_weights(model_scores)
        
        return {
            'models': trained_models,
            'scores': model_scores,
            'weights': ensemble_weights,
            'features': list(features.columns)
        }
    
    def _calculate_ensemble_weights(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance."""
        weights = {}
        total_score = 0
        
        # Use CV scores for weighting
        for name, scores in model_scores.items():
            cv_score = max(scores['cv_mean'], 0.01)  # Avoid negative weights
            weights[name] = cv_score
            total_score += cv_score
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_score
        
        return weights
    
    def make_ensemble_prediction(self, models: Dict, weights: Dict, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for name, model in models.items():
            try:
                pred = model.predict(features)
                weighted_pred = pred * weights.get(name, 0)
                predictions.append(weighted_pred)
            except Exception as e:
                logger.error(f"Error making prediction with {name}: {e}")
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(features))

# Initialize optimizers
performance_optimizer = PerformanceOptimizer()
ml_supercharger = MLAISupercharger()

# Performance monitoring
@performance_optimizer.enhanced_cache(ttl=300)
def get_optimized_market_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Get optimized market data with caching."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return pd.DataFrame()
        
        # Optimize memory usage
        data = performance_optimizer.optimize_dataframe(data)
        
        # Add AI-enhanced features
        data = ml_supercharger.create_advanced_features(data)
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def monitor_performance():
    """Monitor application performance."""
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    if memory_usage > 80:
        st.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
    if cpu_usage > 80:
        st.warning(f"⚠️ High CPU usage: {cpu_usage:.1f}%")
    
    # Display performance metrics in sidebar
    with st.sidebar:
        st.markdown("### 📊 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory", f"{memory_usage:.1f}%")
        with col2:
            st.metric("CPU", f"{cpu_usage:.1f}%")

if __name__ == "__main__":
    logger.info("Performance optimizer and ML supercharger initialized")