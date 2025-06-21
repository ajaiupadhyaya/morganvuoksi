"""Streamlit dashboard for portfolio and model monitoring."""
import pandas as pd
import streamlit as st

from src.execution.options_engine import OptionsEngine
from fundamentals.dcf import discounted_cash_flow


def load_trades(path: str = "option_trades.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def main():
    st.title("Quant Dashboard")
    st.sidebar.header("Actions")
    view = st.sidebar.selectbox("View", ["Trades", "DCF"])

    if view == "Trades":
        df = load_trades()
        st.subheader("Trade History")
        st.dataframe(df)
    else:
        symbol = st.text_input("Symbol", "AAPL")
        if st.button("Run DCF"):
            result = discounted_cash_flow(symbol)
            st.write(result)


if __name__ == "__main__":
    main()
