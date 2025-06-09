"""
Tests for the data fetcher module.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
import pandas as pd

from src.data.fetcher import DataFetcher

@pytest.fixture
def fetcher():
    """Create a DataFetcher instance for testing."""
    return DataFetcher()

@pytest.mark.asyncio
async def test_fetch_stock_data(fetcher):
    """Test fetching stock data from yfinance."""
    # Test parameters
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Fetch data
    data = await fetcher.fetch_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        source='yfinance'
    )
    
    # Verify data
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert data.index.is_monotonic_increasing
    assert data.index.is_monotonic_decreasing is False
    
    # Validate data quality
    assert fetcher.validate_data(data)

@pytest.mark.asyncio
async def test_fetch_market_data(fetcher):
    """Test fetching market data for multiple symbols."""
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Fetch data
    data_dict = await fetcher.fetch_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify data
    assert isinstance(data_dict, dict)
    assert len(data_dict) == len(symbols)
    
    for symbol, data in data_dict.items():
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert data.index.is_monotonic_increasing
        assert data.index.is_monotonic_decreasing is False
        
        # Validate data quality
        assert fetcher.validate_data(data)

@pytest.mark.asyncio
async def test_fetch_economic_data(fetcher):
    """Test fetching economic data from FRED."""
    # Test parameters
    series_ids = ['GDP', 'UNRATE', 'CPIAUCSL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch data
    data_dict = await fetcher.fetch_economic_data(
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify data
    assert isinstance(data_dict, dict)
    assert len(data_dict) == len(series_ids)
    
    for series_id, data in data_dict.items():
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert series_id in data.columns
        assert data.index.is_monotonic_increasing
        assert data.index.is_monotonic_decreasing is False 