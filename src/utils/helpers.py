# src/utils/helpers.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import Union

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

def format_number(value: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes (K, M, B, T)."""
    if pd.isna(value) or value == 0:
        return "0"
    
    value = float(value)
    
    if abs(value) >= 1e12:
        return f"{value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.2f}"

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for OHLCV data."""
    data = df.copy()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + 2 * rolling_std
    data['BB_Lower'] = data['SMA_20'] - 2 * rolling_std
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
    
    return data
