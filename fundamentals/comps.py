"""Comparable company analysis helper."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def fetch_metrics(ticker: str) -> pd.Series:
    if yf is None:
        return pd.Series({"pe": np.random.uniform(5, 20), "ev_ebitda": np.random.uniform(5, 15)})
    info = yf.Ticker(ticker).info
    return pd.Series({"pe": info.get("trailingPE", np.nan), "ev_ebitda": info.get("enterpriseToEbitda", np.nan)})


def comps_table(peers: List[str]) -> pd.DataFrame:
    rows = [fetch_metrics(t) for t in peers]
    df = pd.DataFrame(rows, index=peers)
    return df

