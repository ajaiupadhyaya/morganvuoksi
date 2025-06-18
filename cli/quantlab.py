"""Command line interface for common tasks."""
import argparse
from pathlib import Path
import pandas as pd

from src.data.fetcher import DataFetcher
from src.signals.ml_models import ModelLoader
from src.execution.options_engine import OptionsEngine
from fundamentals.dcf import discounted_cash_flow


def fetch_data(args):
    fetcher = DataFetcher()
    data = awaitable(fetcher.fetch_stock_data(args.symbol, args.start, args.end))
    print(data.head())


def train_model(args):
    loader = ModelLoader()
    model = loader.get(args.model)
    df = pd.read_csv(args.data)
    if hasattr(model, 'fit'):
        y = df.pop('target')
        model.fit(df, y)


def run_backtest(args):
    print("Backtest placeholder")


def optimize_portfolio(args):
    print("Portfolio optimization placeholder")


def build_dcf(args):
    result = discounted_cash_flow(args.symbol)
    print(result)


def live_trade(args):
    engine = OptionsEngine(paper=True)
    engine.enter_trade(args.symbol, 'call', args.strike, args.expiry)
    print(engine.get_history())


def generate_report(args):
    print("Report generation placeholder")


def awaitable(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


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
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
