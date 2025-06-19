"""Unified CLI for MorganVuoksi pipelines."""
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv

from src.data.fetcher import DataFetcher
from src.signals.ml_models import ModelLoader
from src.execution.options_engine import OptionsEngine
from fundamentals.dcf import discounted_cash_flow
from src.utils.logging import setup_logger

CONFIG_FILE = Path("config/config.yaml")
load_dotenv()
logger = setup_logger("quantlab")


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file if present."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}


def fetch_data(args) -> None:
    """Fetch market data for a single symbol."""
    fetcher = DataFetcher()
    data = asyncio.run(
        fetcher.fetch_stock_data(args.symbol, pd.to_datetime(args.start), pd.to_datetime(args.end))
    )
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.symbol}_data.csv"
    data.to_csv(out_path)
    logger.info(f"Saved data to {out_path}")


def train_model(args) -> None:
    """Train a specified model on a CSV dataset."""
    loader = ModelLoader()
    model = loader.get(args.model)
    df = pd.read_csv(args.data)
    if hasattr(model, "fit"):
        y = df.pop("target")
        model.fit(df, y)
        logger.info("Model trained")


def run_backtest(args) -> None:
    """Run backtest using the provided dataset and model."""
    logger.info("Running backtest (simplified)")
    df = pd.read_csv(args.data)
    returns = df["Close"].pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / "backtest.csv"
    cumulative.to_csv(out_path)
    logger.info(f"Backtest saved to {out_path}")


def optimize_portfolio(args) -> None:
    """Optimize portfolio weights based on historical returns."""
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    cov = returns.cov() * 252
    mean = returns.mean() * 252
    inv = np.linalg.pinv(cov.values)
    weights = inv.dot(mean.values)
    weights = weights / weights.sum()
    w = dict(zip(returns.columns, weights))
    logger.info(f"Optimal weights: {w}")


def build_dcf(args) -> None:
    """Run a DCF valuation for the specified symbol."""
    result = discounted_cash_flow(args.symbol)
    logger.info(f"DCF present value: {result.present_value:.2f}")


def live_trade(args) -> None:
    """Execute a single paper trade."""
    engine = OptionsEngine(paper=True)
    engine.enter_trade(args.symbol, "call", args.strike, args.expiry)
    logger.info(engine.get_history().tail(1))


def generate_report(args) -> None:
    """Generate a simple HTML report from trade history."""
    trades = OptionsEngine(paper=True).get_history()
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / "trades.html"
    trades.to_html(out_path, index=False)
    logger.info(f"Report saved to {out_path}")




def main():
    parser = argparse.ArgumentParser(description="QuantLab CLI")
    sub = parser.add_subparsers(dest="command")

    fd = sub.add_parser('fetch-data')
    fd.add_argument('--symbol')
    fd.add_argument('--start')
    fd.add_argument('--end')
    fd.set_defaults(func=fetch_data)

    tm = sub.add_parser('train-model')
    tm.add_argument('--model')
    tm.add_argument('--data')
    tm.set_defaults(func=train_model)

    sub.add_parser('run-backtest').set_defaults(func=run_backtest)
    sub.add_parser('optimize-portfolio').set_defaults(func=optimize_portfolio)

    dcf = sub.add_parser('build-dcf')
    dcf.add_argument('--symbol')
    dcf.set_defaults(func=build_dcf)

    lt = sub.add_parser('live-trade')
    lt.add_argument('--symbol')
    lt.add_argument('--strike', type=float)
    lt.add_argument('--expiry')
    lt.set_defaults(func=live_trade)

    sub.add_parser('generate-report').set_defaults(func=generate_report)

    args = parser.parse_args()
    _ = load_config()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == '__main__':
    main()
