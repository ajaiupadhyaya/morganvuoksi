import pandas as pd
from src.data.market_data import MarketDataLoader

def test_market_data_loader_fetch(tmp_path):
    loader = MarketDataLoader(cache_dir=tmp_path)
    df = loader.fetch("AAPL", "2023-01-03", "2023-01-10")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"Close", "rsi", "macd", "sma_20"}.issubset(df.columns)

