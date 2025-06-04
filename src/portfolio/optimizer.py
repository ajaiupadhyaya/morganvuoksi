# src/portfolio/optimizer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

INPUT_PATH = "data/processed/"
OUTPUT_PATH = "data/processed/"
REPORT_PATH = "outputs/reports/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# ========== UTILITY FUNCTIONS ==========

def load_returns():
    df = pd.read_csv(os.path.join(INPUT_PATH, "returns_combined.csv"), parse_dates=["Date"])
    pivot = df.pivot(index="Date", columns="symbol", values="Return").dropna()
    return pivot

def expected_returns(returns_df):
    return returns_df.mean() * 252  # annualized

def covariance_matrix(returns_df):
    return returns_df.cov() * 252  # annualized

# ========== MEAN-VARIANCE OPTIMIZATION ==========

def mean_variance_optimizer(mu, cov, allow_short=False):
    num_assets = len(mu)

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov @ weights)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * num_assets

    result = minimize(portfolio_volatility,
                      x0=np.ones(num_assets)/num_assets,
                      bounds=bounds,
                      constraints=constraints)

    return pd.Series(result.x, index=mu.index)

# ========== RISK PARITY ALLOCATION ==========

def risk_parity_weights(cov):
    inv_vol = 1 / np.sqrt(np.diag(cov))
    weights = inv_vol / inv_vol.sum()
    return pd.Series(weights, index=cov.columns)

# ========== CVaR OPTIMIZATION (Simple Approximation) ==========

def cvar_objective(weights, returns, alpha=0.05):
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = -portfolio_returns[portfolio_returns < var].mean()
    return cvar

def cvar_optimizer(returns_df, alpha=0.05):
    num_assets = returns_df.shape[1]
    bounds = [(0.0, 1.0)] * num_assets
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(
        cvar_objective,
        x0=np.ones(num_assets)/num_assets,
        args=(returns_df, alpha),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return pd.Series(result.x, index=returns_df.columns)

# ========== VISUALIZATION ==========

def plot_allocation(weights, title):
    plt.figure(figsize=(7, 7))
    weights.plot.pie(autopct='%1.1f%%')
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(REPORT_PATH, filename))
    plt.close()

# ========== MAIN EXECUTION ==========

def main():
    returns_df = load_returns()
    mu = expected_returns(returns_df)
    cov = covariance_matrix(returns_df)

    # === 1. Mean-Variance Optimization
    w_mvo = mean_variance_optimizer(mu, cov, allow_short=False)
    plot_allocation(w_mvo, "MVO Portfolio Weights")
    w_mvo.to_csv(os.path.join(OUTPUT_PATH, "portfolio_weights_mvo.csv"))

    # === 2. Risk-Parity
    w_rp = risk_parity_weights(cov)
    plot_allocation(w_rp, "Risk-Parity Weights")
    w_rp.to_csv(os.path.join(OUTPUT_PATH, "portfolio_weights_risk_parity.csv"))

    # === 3. CVaR Optimizer
    w_cvar = cvar_optimizer(returns_df, alpha=0.05)
    plot_allocation(w_cvar, "CVaR-Optimal Weights")
    w_cvar.to_csv(os.path.join(OUTPUT_PATH, "portfolio_weights_cvar.csv"))

    print("[✓] Portfolio weights saved for MVO, Risk Parity, and CVaR.")

if __name__ == "__main__":
    main()