# src/signals/alpha_factors.py
from src.utils.helpers import fetch_price_data, save_alpha_factors
import pandas as pd

def calculate_factors(prices: pd.DataFrame) -> pd.DataFrame:
    print("📈 Calculating alpha factors...")
    prices = prices.sort_values(by=["symbol", "date"])
    factors = []

    for symbol, group in prices.groupby("symbol"):
        group = group.copy()
        group["ZScore"] = (group["close"] - group["close"].rolling(20).mean()) / group["close"].rolling(20).std()
        group["Momentum_10D"] = group["close"].pct_change(10)
        group["MA_Crossover"] = group["close"] - group["close"].rolling(50).mean()
        group["Bollinger_Width"] = (group["close"].rolling(20).std() / group["close"].rolling(20).mean()) * 100
        delta = group["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        group["RSI"] = 100 - (100 / (1 + rs))

        factors.append(group[["date", "symbol", "ZScore", "Momentum_10D", "MA_Crossover", "Bollinger_Width", "RSI"]])

    return pd.concat(factors, ignore_index=True)

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    prices = fetch_price_data(tickers)
    factors = calculate_factors(prices)
    save_alpha_factors(factors, "alpha_factors")

if __name__ == "__main__":
    main()
