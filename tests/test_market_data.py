import pytest
from src.data import market_data

def test_load_market_data(sample_market_data):
    # Simulate loading data
    df = sample_market_data
    assert not df.empty
    assert set(['AAPL', 'MSFT', 'GOOG']).issubset(df.columns)

def test_data_validation(sample_market_data):
    # Example: check for NaNs
    df = sample_market_data
    assert df.isnull().sum().sum() == 0 
