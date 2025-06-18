__all__ = ["DataFetcher", "MarketDataLoader", "load_market_data"]

def __getattr__(name):
    if name == "DataFetcher":
        from .fetcher import DataFetcher
        return DataFetcher
    if name in {"MarketDataLoader", "load_market_data"}:
        from .market_data import MarketDataLoader, load_market_data
        return locals()[name]
    raise AttributeError(name)
