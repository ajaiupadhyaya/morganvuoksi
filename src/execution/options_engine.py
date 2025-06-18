"""Simplified options trading engine supporting simulation and paper trading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - network
    yf = None


@dataclass
class OptionTrade:
    symbol: str
    option_type: str
    strike: float
    expiry: str
    qty: int
    price: float
    greek_delta: float
    timestamp: pd.Timestamp


class OptionsEngine:
    """Execute option trades with basic greeks-based sizing and slippage."""

    def __init__(self, paper: bool = True, log_path: str = "option_trades.csv") -> None:
        self.paper = paper
        self.log_path = Path(log_path)
        self.trades: List[OptionTrade] = []

    def _get_option_price(self, symbol: str, strike: float, expiry: str, option_type: str) -> float:
        if yf is None:
            return 1.0  # fallback price
        opt = yf.Ticker(symbol).option_chain(expiry)
        chain = opt.calls if option_type.lower() == "call" else opt.puts
        row = chain.loc[(chain["strike"] == strike)]
        if row.empty:
            return 1.0
        return float(row.iloc[0]["lastPrice"])

    def _slippage(self, price: float) -> float:
        return price * np.random.uniform(0.999, 1.001)

    def _size_from_delta(self, target_delta: float, option_delta: float) -> int:
        if option_delta == 0:
            return 0
        return int(target_delta / option_delta)

    def enter_trade(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry: str,
        target_delta: float = 0.1,
    ) -> OptionTrade:
        price = self._get_option_price(symbol, strike, expiry, option_type)
        option_delta = 0.5  # placeholder
        qty = self._size_from_delta(target_delta, option_delta)
        exec_price = self._slippage(price)

        trade = OptionTrade(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            qty=qty,
            price=exec_price,
            greek_delta=option_delta,
            timestamp=pd.Timestamp.utcnow(),
        )
        self.trades.append(trade)
        self._log_trade(trade)
        return trade

    def _log_trade(self, trade: OptionTrade) -> None:
        df = pd.DataFrame([trade.__dict__])
        if self.log_path.exists():
            df_prev = pd.read_csv(self.log_path)
            df = pd.concat([df_prev, df], ignore_index=True)
        df.to_csv(self.log_path, index=False)

    def get_history(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.log_path)

