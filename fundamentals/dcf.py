"""Discounted Cash Flow valuation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@dataclass
class DCFResult:
    present_value: float
    assumptions: Dict


def fetch_cashflows(symbol: str) -> pd.DataFrame:
    if yf is None:
        dates = pd.date_range("2018", periods=5, freq="Y")
        return pd.DataFrame({"fcf": np.random.uniform(1e8, 1e9, len(dates))}, index=dates)
    ticker = yf.Ticker(symbol)
    cf = ticker.cashflow
    if cf.empty:
        dates = pd.date_range("2018", periods=5, freq="Y")
        cf = pd.DataFrame({"fcf": np.random.uniform(1e8, 1e9, len(dates))}, index=dates)
    else:
        cf = cf.T
        cf.index = pd.to_datetime(cf.index)
        cf = cf[["Free Cash Flow"]].rename(columns={"Free Cash Flow": "fcf"})
    return cf.sort_index()


def discounted_cash_flow(symbol: str, wacc: float = 0.1, terminal_growth: float = 0.02) -> DCFResult:
    cf = fetch_cashflows(symbol)
    years = np.arange(1, len(cf) + 1)
    discounts = 1 / (1 + wacc) ** years
    pv = np.sum(cf["fcf"].values * discounts)
    terminal = cf["fcf"].iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    terminal_pv = terminal / (1 + wacc) ** len(cf)
    total_value = pv + terminal_pv
    return DCFResult(total_value, {"wacc": wacc, "terminal_growth": terminal_growth})

