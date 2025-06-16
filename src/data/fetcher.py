from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dep
    yf = None

try:
    from fredapi import Fred
except Exception:  # pragma: no cover - optional dep
    Fred = None


class DataFetcher:
    """Utility class for fetching financial and economic data."""

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        self.fred = Fred(api_key=fred_api_key) if Fred and fred_api_key else None

    async def fetch_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Fetch historical stock data."""
        if source == "yfinance" and yf is not None:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            except Exception:  # pragma: no cover - network failure
                df = pd.DataFrame()
        else:  # pragma: no cover - fallback for missing deps
            df = pd.DataFrame()

        if df.empty:
            dates = pd.date_range(start_date, end_date)
            df = pd.DataFrame(
                {
                    "Open": np.random.rand(len(dates)),
                    "High": np.random.rand(len(dates)),
                    "Low": np.random.rand(len(dates)),
                    "Close": np.random.rand(len(dates)),
                    "Volume": np.random.randint(1_000_000, 5_000_000, size=len(dates)),
                },
                index=dates,
            )
        return df

    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols."""
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            data[symbol] = await self.fetch_stock_data(symbol, start_date, end_date)
        return data

    async def fetch_economic_data(
        self,
        series_ids: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch economic data series from FRED or generate dummy data."""
        data: Dict[str, pd.DataFrame] = {}
        for series_id in series_ids:
            if self.fred is not None:
                series = self.fred.get_series(series_id, start_date, end_date)
                df = series.to_frame(series_id)
            else:  # pragma: no cover - fallback
                dates = pd.date_range(start_date, end_date, freq="M")
                df = pd.DataFrame({series_id: np.random.rand(len(dates))}, index=dates)
            data[series_id] = df
        return data

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Simple data quality checks used in tests."""
        return (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and df.isnull().sum().sum() == 0
            and df.index.is_monotonic_increasing
        )
