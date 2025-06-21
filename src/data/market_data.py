"""
Market Data Fetcher
Integrates multiple data sources: Alpaca, Polygon, Yahoo Finance, FRED
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data sources."""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    polygon_api_key: str = ""
    fred_api_key: str = ""
    yahoo_finance_enabled: bool = True
    cache_duration: int = 300  # 5 minutes

class MarketDataFetcher:
    """Comprehensive market data fetcher with multiple sources."""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.session = requests.Session()
        self.cache = {}
        self.cache_timestamps = {}
        
        # Load API keys from environment
        self._load_api_keys()
        
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        if not self.config.alpaca_api_key:
            self.config.alpaca_api_key = os.getenv('ALPACA_API_KEY', '')
        if not self.config.alpaca_secret_key:
            self.config.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        if not self.config.polygon_api_key:
            self.config.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        if not self.config.fred_api_key:
            self.config.fred_api_key = os.getenv('FRED_API_KEY', '')
    
    def get_stock_data(self, symbol: str, period: str = "1y", 
                      interval: str = "1d", source: str = "yahoo") -> pd.DataFrame:
        """Get stock data from specified source."""
        cache_key = f"{symbol}_{period}_{interval}_{source}"
        
        # Check cache
        if cache_key in self.cache:
            if time.time() - self.cache_timestamps[cache_key] < self.config.cache_duration:
                return self.cache[cache_key]
        
        try:
            if source == "yahoo" and self.config.yahoo_finance_enabled:
                data = self._get_yahoo_data(symbol, period, interval)
            elif source == "alpaca" and self.config.alpaca_api_key:
                data = self._get_alpaca_data(symbol, period, interval)
            elif source == "polygon" and self.config.polygon_api_key:
                data = self._get_polygon_data(symbol, period, interval)
            else:
                # Fallback to Yahoo Finance
                data = self._get_yahoo_data(symbol, period, interval)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the result
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = time.time()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_yahoo_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.title() for col in data.columns]
            data.index.name = 'Date'
            
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_alpaca_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Alpaca."""
        try:
            headers = {
                'APCA-API-KEY-ID': self.config.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.config.alpaca_secret_key
            }
            
            # Convert period to date range
            end_date = datetime.now()
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "5d":
                start_date = end_date - timedelta(days=5)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            elif period == "5y":
                start_date = end_date - timedelta(days=1825)
            elif period == "10y":
                start_date = end_date - timedelta(days=3650)
            else:
                start_date = end_date - timedelta(days=365)
            
            url = f"{self.config.alpaca_base_url}/v2/stocks/{symbol}/bars"
            params = {
                'timeframe': interval,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'limit': 10000
            }
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'bars' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars = data['bars']
            df = pd.DataFrame(bars)
            df['t'] = pd.to_datetime(df['t'])
            df.set_index('t', inplace=True)
            df.index.name = 'Date'
            
            # Rename columns
            df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'n': 'TradeCount',
                'vw': 'VWAP'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Alpaca error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_polygon_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Polygon."""
        try:
            # Convert period to date range
            end_date = datetime.now()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Convert interval to Polygon format
            interval_map = {
                "1d": "day",
                "1h": "hour",
                "5m": "5minute",
                "1m": "minute"
            }
            polygon_interval = interval_map.get(interval, "day")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{polygon_interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apiKey': self.config.polygon_api_key,
                'limit': 50000
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'OK' or 'results' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            results = data['results']
            df = pd.DataFrame(results)
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('t', inplace=True)
            df.index.name = 'Date'
            
            # Rename columns
            df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'n': 'TradeCount',
                'vw': 'VWAP'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        if data.empty:
            return data
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = self._calculate_macd(data['Close'])
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Price momentum
        data['Momentum'] = data['Close'].pct_change(periods=10)
        data['ROC'] = (data['Close'] / data['Close'].shift(10) - 1) * 100
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def get_market_data(self, symbols: List[str], period: str = "1y", 
                       interval: str = "1d", source: str = "yahoo") -> Dict[str, pd.DataFrame]:
        """Get market data for multiple symbols."""
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_stock_data(symbol, period, interval, source)
        return data
    
    def get_economic_data(self, series_ids: List[str], start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """Get economic data from FRED."""
        if not self.config.fred_api_key:
            logger.warning("FRED API key not configured")
            return pd.DataFrame()
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            data = {}
            for series_id in series_ids:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.config.fred_api_key,
                    'file_type': 'json',
                    'observation_start': start_date,
                    'observation_end': end_date
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                result = response.json()
                if 'observations' in result:
                    observations = result['observations']
                    dates = [obs['date'] for obs in observations]
                    values = [float(obs['value']) if obs['value'] != '.' else np.nan for obs in observations]
                    
                    series_data = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
                    data[series_id] = series_data
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"FRED API error: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol: str, period: str = "1y", 
                       interval: str = "1d") -> pd.DataFrame:
        """Get cryptocurrency data."""
        # Use Yahoo Finance for crypto
        crypto_symbol = f"{symbol}-USD"
        return self.get_stock_data(crypto_symbol, period, interval, "yahoo")
    
    def get_forex_data(self, pair: str, period: str = "1y", 
                      interval: str = "1d") -> pd.DataFrame:
        """Get forex data."""
        # Use Yahoo Finance for forex
        forex_symbol = f"{pair}=X"
        return self.get_stock_data(forex_symbol, period, interval, "yahoo")
    
    def get_options_chain(self, symbol: str, expiration_date: str = None) -> Dict:
        """Get options chain data."""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.options
            
            if not options:
                return {}
            
            # Use first available expiration if none specified
            if not expiration_date:
                expiration_date = options[0]
            
            chain = ticker.option_chain(expiration_date)
            
            return {
                'calls': chain.calls,
                'puts': chain.puts,
                'expiration': expiration_date,
                'available_expirations': options
            }
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return {}
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'volume_avg': info.get('averageVolume', 0),
                'price_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'profit_margins': info.get('profitMargins', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment indicators."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get institutional holders
            institutional_holders = ticker.institutional_holders
            major_holders = ticker.major_holders
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            
            # Get earnings dates
            calendar = ticker.calendar
            
            return {
                'institutional_holders': institutional_holders.to_dict('records') if institutional_holders is not None else [],
                'major_holders': major_holders.to_dict('records') if major_holders is not None else [],
                'recommendations': recommendations.to_dict('records') if recommendations is not None else [],
                'earnings_calendar': calendar.to_dict('records') if calendar is not None else []
            }
            
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
    
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        return {
            'cache_size': len(self.cache),
            'cached_symbols': list(self.cache.keys()),
            'oldest_cache': min(self.cache_timestamps.values()) if self.cache_timestamps else 0,
            'newest_cache': max(self.cache_timestamps.values()) if self.cache_timestamps else 0
        }

# Common economic indicators
ECONOMIC_INDICATORS = {
    'GDP': 'GDP',
    'Unemployment': 'UNRATE',
    'Inflation': 'CPIAUCSL',
    'Federal_Funds_Rate': 'FEDFUNDS',
    '10Y_Treasury': 'GS10',
    '2Y_Treasury': 'GS2',
    'VIX': 'VIXCLS',
    'Dollar_Index': 'DTWEXBGS',
    'Oil_Price': 'DCOILWTICO',
    'Gold_Price': 'GOLDPMGBD228NLBM'
}

# Sector ETFs for market analysis
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate'
}
