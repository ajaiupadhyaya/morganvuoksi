"""Automated pipeline invoking the QuantLab CLI functions."""
from argparse import Namespace

from cli.quantlab import (
    fetch_data,
    train_model,
    run_backtest,
    optimize_portfolio,
    build_dcf,
    live_trade,
    generate_report,
)


def main() -> None:
    # Sample workflow using built-in defaults
    fetch_data(Namespace(symbol="AAPL", start="2024-01-01", end="2024-03-01"))
    train_model(Namespace(model="xgboost", data="data/AAPL_data.csv"))
    run_backtest(Namespace(data="data/AAPL_data.csv"))
    optimize_portfolio(Namespace(data="data/AAPL_data.csv"))
    build_dcf(Namespace(symbol="AAPL"))
    live_trade(Namespace(symbol="AAPL", strike=150.0, expiry="2025-06-20"))
    generate_report(Namespace())


if __name__ == "__main__":
    main()
