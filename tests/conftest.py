import pytest
import pandas as pd
import numpy as np
from src.config import Config

@pytest.fixture(scope='session')
def config():
    return Config()

@pytest.fixture
def sample_market_data():
    # Example DataFrame for market data
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'AAPL': np.random.rand(5),
        'MSFT': np.random.rand(5),
        'GOOG': np.random.rand(5)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_signals():
    # Example DataFrame for signals
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'AAPL': np.random.randn(5),
        'MSFT': np.random.randn(5),
        'GOOG': np.random.randn(5)
    }
    return pd.DataFrame(data, index=dates) 