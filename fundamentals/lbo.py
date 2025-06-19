"""Leverage Buyout valuation helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy_financial as npf


@dataclass
class LBOResult:
    irr: float
    cash_flows: List[float]


def lbo_model(
    entry_equity: float,
    debt: float,
    interest_rate: float,
    cash_flow_growth: float,
    exit_multiple: float,
    years: int = 5,
) -> LBOResult:
    equity = entry_equity
    cf = []
    for year in range(1, years + 1):
        ebitda = equity * (1 + cash_flow_growth) ** year
        interest = debt * interest_rate
        cash_flow = ebitda - interest
        cf.append(cash_flow)
    exit_value = cf[-1] * exit_multiple
    cf[-1] += exit_value
    irr = npf.irr([-entry_equity] + cf)
    return LBOResult(irr, cf)

