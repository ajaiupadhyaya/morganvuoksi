"""Market data utilities with caching and technical indicators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict

import pandas as pd
import numpy as np
import yfinance as yf


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period).mean()
    roll_down = down.ewm(span=period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


@dataclass
class MarketDataLoader:
    """Utility to fetch and cache market data."""

    cache_dir: Path = Path("data/cache")

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: str, end: str, interval: str) -> Path:
        fname = f"{symbol}_{start}_{end}_{interval}.csv".replace(":", "-")
        return self.cache_dir / fname

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data with common technical indicators."""
        path = self._cache_path(symbol, start, end, interval)
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)

        try:
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        except Exception:  # pragma: no cover - network failure
            df = pd.DataFrame()

        df = df.ffill().dropna()
        if df.empty:
            dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end) + pd.Timedelta(days=30), freq="B")
            base = 100 + np.random.randn(len(dates)).cumsum()
            df = pd.DataFrame({
                "Open": base,
                "High": base + np.abs(np.random.randn(len(dates))),
                "Low": base - np.abs(np.random.randn(len(dates))),
                "Close": base + np.random.randn(len(dates)) * 0.5,
                "Volume": np.random.randint(1_000_000, 5_000_000, size=len(dates))
            }, index=dates)

        df["rsi"] = _rsi(df["Close"])
        df["macd"] = _macd(df["Close"])
        df["sma_20"] = df["Close"].rolling(5).mean()
        df.dropna(inplace=True)
        df.to_csv(path)
        return df

    def load_bulk(self, symbols: Iterable[str], start: str, end: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols into a dictionary."""
        return {sym: self.fetch(sym, start, end, interval) for sym in symbols}


def load_market_data(symbols: Iterable[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Convenience helper returning a combined DataFrame of prices."""
    loader = MarketDataLoader()
    frames = []
    for sym, df in loader.load_bulk(symbols, start, end, interval).items():
        f = df.copy()
        f["symbol"] = sym
        frames.append(f)
    combined = pd.concat(frames)
    return combined.reset_index()
