# src/portfolio/optimizer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import cvxpy as cp
import seaborn as sns
from src.config import config
from dataclasses import dataclass
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
import warnings
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

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
    """Main execution function."""
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizer()
        
        # Load data
        returns_path = config.paths.processed_data_dir / "processed_data_20240315_1200.parquet"
        returns_df = pd.read_parquet(returns_path)
        
        # Run different optimization strategies
        strategies = {
            'Mean-Variance': optimizer.optimize_portfolio(returns_df, method="mean_variance"),
            'Risk Parity': optimizer.optimize_portfolio(returns_df, method="risk_parity"),
            'CVaR': optimizer.optimize_portfolio(returns_df, method="cvar")
        }
        
        # Plot and save results
        for name, strategy in strategies.items():
            optimizer.plot_optimization_results(strategy['weights'], returns_df, name)
            optimizer.save_weights(strategy['weights'], f"{name.lower().replace(' ', '_')}_weights.parquet")
        
        logger.info("Portfolio optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        raise

class PortfolioOptimizationError(Exception):
    """Custom exception for portfolio optimization errors."""
    pass

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 0.2
    max_leverage: float = 1.0
    max_turnover: float = 0.1
    max_sector_exposure: float = 0.3
    max_factor_exposure: float = 0.5
    min_liquidity: float = 1e6
    max_volatility: float = 0.2
    max_drawdown: float = 0.15
    min_diversification: float = 0.5
    min_factor_exposure: float = 0.1
    max_correlation: float = 0.7
    min_holding_period: int = 5
    max_holding_period: int = 20

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_position_size: float = 0.3
    min_position_size: float = 0.0
    max_sector_exposure: float = 0.4
    max_factor_exposure: float = 0.3
    transaction_costs: float = 0.001
    rebalance_frequency: str = 'monthly'

class PortfolioOptimizer:
    """Advanced portfolio optimizer with multiple strategies."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = "mean_variance", 
                          risk_tolerance: str = "moderate", **kwargs) -> Dict:
        """Main portfolio optimization method."""
        
        if method == "mean_variance":
            return self._mean_variance_optimization(returns, risk_tolerance)
        elif method == "black_litterman":
            return self._black_litterman_optimization(returns, **kwargs)
        elif method == "risk_parity":
            return self._risk_parity_optimization(returns)
        elif method == "maximum_sharpe":
            return self._maximum_sharpe_optimization(returns)
        elif method == "minimum_variance":
            return self._minimum_variance_optimization(returns)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _mean_variance_optimization(self, returns: pd.DataFrame, 
                                   risk_tolerance: str) -> Dict:
        """Mean-variance optimization."""
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Risk tolerance mapping
        risk_tolerance_map = {
            "conservative": 0.1,
            "moderate": 0.2,
            "aggressive": 0.3
        }
        
        target_volatility = risk_tolerance_map.get(risk_tolerance, 0.2)
        
        # Optimization constraints
        n_assets = len(returns.columns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            return -sharpe_ratio
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: target_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}  # Volatility constraint
        ]
        
        # Bounds
        bounds = [(self.config.min_position_size, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess (equal weight)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            
            return {
                'weights': pd.Series(optimal_weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'mean_variance',
                'risk_tolerance': risk_tolerance
            }
        else:
            raise ValueError("Optimization failed")
    
    def _black_litterman_optimization(self, returns: pd.DataFrame, 
                                    views: Dict = None, 
                                    view_confidences: Dict = None) -> Dict:
        """Black-Litterman optimization with investor views."""
        
        # Calculate market equilibrium returns
        market_caps = self._get_market_caps(returns.columns)  # Placeholder
        if market_caps is None:
            market_caps = pd.Series([1/len(returns.columns)] * len(returns.columns), 
                                   index=returns.columns)
        
        cov_matrix = returns.cov() * 252
        risk_aversion = 3.0  # Typical value
        
        # Market equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_caps)
        
        # Default views (can be overridden)
        if views is None:
            views = {}
        
        if view_confidences is None:
            view_confidences = {}
        
        # Create view matrix
        P = np.zeros((len(views), len(returns.columns)))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (asset, view) in enumerate(views.items()):
            if asset in returns.columns:
                asset_idx = returns.columns.get_loc(asset)
                P[i, asset_idx] = 1
                Q[i] = view
                Omega[i, i] = view_confidences.get(asset, 0.1)
        
        # Black-Litterman formula
        tau = 0.05  # Scaling factor
        
        # Posterior covariance
        M = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))
        
        # Posterior returns
        mu = np.dot(M, np.dot(np.linalg.inv(tau * cov_matrix), pi) + 
                   np.dot(P.T, np.dot(np.linalg.inv(Omega), Q)))
        
        # Optimize with posterior estimates
        expected_returns = pd.Series(mu, index=returns.columns)
        
        return self._mean_variance_optimization(returns, "moderate")
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict:
        """Risk parity optimization."""
        
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Objective function: minimize risk contribution differences
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
            target_contribution = portfolio_vol / n_assets
            return np.sum((risk_contributions - target_contribution) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_position_size, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            expected_returns = returns.mean() * 252
            portfolio_return = np.sum(expected_returns * optimal_weights)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            
            return {
                'weights': pd.Series(optimal_weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'risk_parity'
            }
        else:
            raise ValueError("Risk parity optimization failed")
    
    def _maximum_sharpe_optimization(self, returns: pd.DataFrame) -> Dict:
        """Maximum Sharpe ratio optimization."""
        
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            return -sharpe_ratio
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_position_size, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            
            return {
                'weights': pd.Series(optimal_weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'maximum_sharpe'
            }
        else:
            raise ValueError("Maximum Sharpe optimization failed")
    
    def _minimum_variance_optimization(self, returns: pd.DataFrame) -> Dict:
        """Minimum variance optimization."""
        
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_position_size, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            expected_returns = returns.mean() * 252
            portfolio_return = np.sum(expected_returns * optimal_weights)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            
            return {
                'weights': pd.Series(optimal_weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'minimum_variance'
            }
        else:
            raise ValueError("Minimum variance optimization failed")
    
    def _get_market_caps(self, symbols: List[str]) -> Optional[pd.Series]:
        """Get market capitalizations for symbols."""
        # This would typically fetch from a data provider
        # For now, return None to use equal weights
        return None
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame, 
                                   num_portfolios: int = 100) -> pd.DataFrame:
        """Calculate efficient frontier."""
        
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Generate random portfolios
        portfolios = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            
            portfolios.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'weights': weights
            })
        
        return pd.DataFrame(portfolios)
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                  weights: pd.Series) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        
        # Basic metrics
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Beta calculation (assuming market returns available)
        # For now, use portfolio volatility as proxy
        beta = annual_vol / 0.15  # Assuming 15% market volatility
        
        # Sector and factor exposures (placeholder)
        sector_exposure = 0.0
        factor_exposure = 0.0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'sector_exposure': sector_exposure,
            'factor_exposure': factor_exposure,
            'weights': weights
        }
    
    def rebalance_portfolio(self, current_weights: pd.Series, 
                           target_weights: pd.Series, 
                           current_prices: pd.Series) -> Dict:
        """Calculate rebalancing trades."""
        
        # Calculate required trades
        trades = target_weights - current_weights
        
        # Calculate transaction costs
        transaction_cost = np.sum(np.abs(trades)) * self.config.transaction_costs
        
        # Filter out small trades (below minimum threshold)
        min_trade_threshold = 0.01  # 1%
        trades_filtered = trades.copy()
        trades_filtered[np.abs(trades) < min_trade_threshold] = 0
        
        # Recalculate transaction costs
        transaction_cost_filtered = np.sum(np.abs(trades_filtered)) * self.config.transaction_costs
        
        return {
            'trades': trades_filtered,
            'transaction_cost': transaction_cost_filtered,
            'trade_count': np.sum(trades_filtered != 0),
            'total_turnover': np.sum(np.abs(trades_filtered))
        }
    
    def plot_optimization_results(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        title: str
    ) -> None:
        """
        Plot optimization results and performance metrics.
        
        Args:
            weights: Series of portfolio weights
            returns: DataFrame of asset returns
            title: Plot title
        """
        try:
            # Calculate metrics
            metrics = self.calculate_portfolio_metrics(returns, weights)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot weights
            weights.plot(kind='bar', ax=ax1)
            ax1.set_title('Portfolio Weights')
            ax1.set_xlabel('Asset')
            ax1.set_ylabel('Weight')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot metrics
            metrics_df = pd.Series(metrics)
            metrics_df.plot(kind='bar', ax=ax2)
            ax2.set_title('Performance Metrics')
            ax2.set_xlabel('Metric')
            ax2.set_ylabel('Value')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(config.paths.outputs_dir / f"{title.lower().replace(' ', '_')}.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise PortfolioOptimizationError(f"Failed to plot results: {str(e)}")
    
    def save_weights(
        self,
        weights: pd.Series,
        filename: Optional[str] = None
    ) -> str:
        """
        Save portfolio weights with proper versioning.
        
        Args:
            weights: Series of portfolio weights
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"portfolio_weights_{timestamp}.parquet"
            
            filepath = config.paths.processed_data_dir / filename
            weights.to_frame('weight').to_parquet(filepath)
            logger.info(f"Portfolio weights saved to {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            raise PortfolioOptimizationError(f"Failed to save weights: {str(e)}")

if __name__ == "__main__":
    main()
