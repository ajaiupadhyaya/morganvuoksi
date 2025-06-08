# src/utils/helpers.py

import yfinance as yf
import pandas as pd
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_price_data(symbols, start="2023-01-01", end="2024-01-01"):
    print(f"📡 Fetching price data for: {symbols}")
    df = yf.download(symbols, start=start, end=end, group_by='ticker', auto_adjust=True)

    if len(symbols) == 1:
        df['symbol'] = symbols[0]
        df = df.reset_index()
    else:
        data = []
        for sym in symbols:
            sym_df = df[sym].copy()
            sym_df['symbol'] = sym
            sym_df = sym_df.reset_index()
            data.append(sym_df)
        df = pd.concat(data)

    df.rename(columns=str.lower, inplace=True)
    return df

def save_alpha_factors(factors: pd.DataFrame, filename: str):
    ensure_dir("data/processed")
    factors.to_parquet(f"data/processed/{filename}.parquet")
    print(f"[✓] Saved: data/processed/{filename}.parquet")