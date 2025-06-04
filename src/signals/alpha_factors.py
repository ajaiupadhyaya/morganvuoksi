# src/signals/alpha_factors.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.helpers import ensure_dir

DATA_PATH = "data/processed/features.csv"
SAVE_PATH = "data/processed/alpha_factors.parquet"

def zscore(df):
    return (df - df.mean()) / df.std()

def compute_factors(df):
    grouped = df.groupby("symbol")
    results = {}

    # 1. Z-Score of Adj Close
    z = grouped["Adj Close"].transform(zscore)
    df["ZScore"] = z
    results["ZScore"] = z

    # 2. 10-Day Momentum
    momentum = grouped["Adj Close"].transform(lambda x: x - x.shift(10))
    df["Momentum_10D"] = momentum
    results["Momentum_10D"] = momentum

    # 3. Moving Average Crossover
    short_ma = grouped["Adj Close"].transform(lambda x: x.rolling(10).mean())
    long_ma = grouped["Adj Close"].transform(lambda x: x.rolling(50).mean())
    ma_crossover = short_ma - long_ma
    df["MA_Crossover"] = ma_crossover
    results["MA_Crossover"] = ma_crossover

    # 4. Bollinger Band Width
    ma20 = grouped["Adj Close"].transform(lambda x: x.rolling(20).mean())
    std20 = grouped["Adj Close"].transform(lambda x: x.rolling(20).std())
    bw = (2 * std20) / ma20
    df["Bollinger_Width"] = bw
    results["Bollinger_Width"] = bw

    # 5. RSI (Relative Strength Index)
    delta = grouped["Adj Close"].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    results["RSI"] = rsi

    return df[["date", "symbol", "ZScore", "Momentum_10D", "MA_Crossover", "Bollinger_Width", "RSI"]]

def save_factor_plots(df):
    ensure_dir("plots/factors")
    for factor in ["ZScore", "Momentum_10D", "MA_Crossover", "Bollinger_Width", "RSI"]:
        pivot = df.pivot(index="date", columns="symbol", values=factor)
        pivot.plot(figsize=(12, 4), title=factor)
        plt.tight_layout()
        plt.savefig(f"plots/factors/{factor}.png")
        plt.close()
        print(f"[✓] Saved: {factor}.png")

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = compute_factors(df)
    ensure_dir("data/processed")
    df.to_parquet(SAVE_PATH)
    print(f"[✓] Saved combined alpha factors to {SAVE_PATH}")
    save_factor_plots(df)

if __name__ == "__main__":
    main()