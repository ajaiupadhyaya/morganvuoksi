import pandas as pd
from fundamentals.dcf import discounted_cash_flow
from fundamentals.lbo import lbo_model
from fundamentals.comps import comps_table


def test_dcf():
    result = discounted_cash_flow('AAPL')
    assert result.present_value > 0


def test_lbo():
    res = lbo_model(100, 50, 0.05, 0.1, 5)
    assert isinstance(res.irr, float)


def test_comps():
    df = comps_table(['AAPL', 'MSFT'])
    assert not df.empty
