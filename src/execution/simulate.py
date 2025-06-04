# src/execution/simulate.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

INPUT_PATH = "data/processed/"
OUTPUT_PATH = "data/processed/"
REPORT_PATH = "outputs/reports/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# ===== CONFIG =====
PORTFOLIO_TYPE = "mvo"      # options: 'mvo', 'risk_parity', 'cvar'
SLIPPAGE_RATE = 0.001       # 0.1% slippage per rebalance
INITIAL_NAV = 1_000_000


# ===== LOAD DATA =====

def load_weights():
    path = os.path.join(INPUT_PATH, f"portfolio_weights_{PORTFOLIO_TYPE}.csv")
    df = pd.read_csv(path)
    return df.set_index("symbol")["0"]

def load_returns():
    df = pd.read_csv(os.path.join(INPUT_PATH, "returns_combined.csv"), parse_dates=["Date"])
    return df.pivot(index="Date", columns="symbol", values="Return").dropna()

def load_benchmark(symbol="SPY"):
    files = sorted(f for f in os.listdir("data/raw") if symbol in f and f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No benchmark file for {symbol} found in data/raw/")
    df = pd.read_csv(os.path.join("data/raw", files[-1]))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Adj Close"]].rename(columns={"Adj Close": symbol})
    df[symbol] = pd.to_numeric(df[symbol], errors="coerce")
    df = df.dropna(subset=[symbol])
    df.set_index("Date", inplace=True)
    df[symbol] = df[symbol].pct_change()
    return df.dropna()


# ===== SIMULATION CORE =====

def simulate_nav(weights: pd.Series, returns: pd.DataFrame, slippage=0.001):
    common_assets = list(set(weights.index) & set(returns.columns))
    weights = weights[common_assets]
    returns = returns[common_assets]

    weights = weights / weights.sum()
    port_returns = (returns @ weights).dropna()

    # Approximate slippage as fixed % of portfolio turnover
    turnover = weights.diff().abs().sum() if not weights.isnull().all() else 0
    port_returns_adj = port_returns - (slippage * turnover)

    nav_series = (1 + port_returns_adj).cumprod() * INITIAL_NAV
    return nav_series, port_returns_adj


# ===== METRICS =====

def compute_metrics(returns, benchmark):
    metrics = {}
    metrics["CAGR"] = (1 + returns.mean())**252 - 1
    metrics["Volatility"] = returns.std() * np.sqrt(252)
    metrics["Sharpe Ratio"] = metrics["CAGR"] / metrics["Volatility"]

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    metrics["Max Drawdown"] = drawdown.min()

    lr = LinearRegression()
    lr.fit(benchmark.values.reshape(-1, 1), returns.values)
    metrics["Beta"] = lr.coef_[0]
    metrics["Alpha"] = returns.mean() * 252 - metrics["Beta"] * benchmark.mean() * 252

    return pd.Series(metrics)


# ===== PLOTTING =====

def plot_nav(nav, benchmark):
    plt.figure(figsize=(12, 6))
    nav.plot(label="Strategy NAV", lw=2)
    benchmark_nav = (1 + benchmark).cumprod() * INITIAL_NAV
    benchmark_nav.plot(label="SPY Benchmark", lw=2)
    plt.title("NAV Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_PATH, "nav_chart.png"))
    plt.close()

def plot_drawdown(returns):
    nav = (1 + returns).cumprod()
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    plt.figure(figsize=(10, 4))
    drawdown.plot(color="red", lw=2)
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_PATH, "drawdown.png"))
    plt.close()


# ===== MAIN EXECUTION =====

def main():
    weights = load_weights()
    returns = load_returns()
    benchmark = load_benchmark()["SPY"]
    benchmark = benchmark.loc[returns.index]

    nav, port_returns = simulate_nav(weights, returns, slippage=SLIPPAGE_RATE)

    metrics = compute_metrics(port_returns, benchmark)
    metrics.to_csv(os.path.join(OUTPUT_PATH, f"performance_metrics_{PORTFOLIO_TYPE}.csv"))
    nav.to_csv(os.path.join(OUTPUT_PATH, f"portfolio_nav_{PORTFOLIO_TYPE}.csv"))

    plot_nav(nav, benchmark)
    plot_drawdown(port_returns)

    print("[✓] NAV + performance simulation complete.")
    print(metrics)

if __name__ == "__main__":
    main()