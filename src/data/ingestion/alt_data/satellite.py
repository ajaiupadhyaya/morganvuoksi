"""Satellite imagery ingestion placeholder returning synthetic data."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def fetch_imagery(ticker: str, dates: List[str] | None = None) -> pd.DataFrame:
    """Return synthetic activity metrics for given dates."""
    dates = dates or ["2024-01-01", "2024-02-01", "2024-03-01"]
    activity = np.random.uniform(0, 1, len(dates))
    df = pd.DataFrame({"date": pd.to_datetime(dates), "activity": activity})
    return df
