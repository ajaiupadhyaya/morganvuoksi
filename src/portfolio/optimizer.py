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
            'Mean-Variance': optimizer.mean_variance_optimize(returns_df),
            'Risk Parity': optimizer.risk_parity_optimize(returns_df),
            'CVaR': optimizer.cvar_optimize(returns_df)
        }
        
        # Plot and save results
        for name, weights in strategies.items():
            optimizer.plot_optimization_results(weights, returns_df, name)
            optimizer.save_weights(weights, f"{name.lower().replace(' ', '_')}_weights.parquet")
        
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

class PortfolioOptimizer:
    """Advanced portfolio optimization with risk management."""
    
    def __init__(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        lookback_period: int = 252,
        n_simulations: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            constraints: Optimization constraints
            lookback_period: Lookback period for calculations
            n_simulations: Number of simulations for CVaR
            random_state: Random state for reproducibility
        """
        self.constraints = constraints or OptimizationConstraints()
        self.lookback_period = lookback_period
        self.n_simulations = n_simulations
        self.random_state = random_state
        
        # Initialize tracking
        self.weights_history = {}
        self.performance_metrics = {}
    
    def calculate_metrics(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        benchmark: Optional[pd.Series] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
            benchmark: Benchmark returns
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Portfolio returns
            portfolio_returns = returns @ weights
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility
            
            # Drawdown metrics
            cum_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Risk metrics
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            downside_dev = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
            sortino_ratio = annual_return / downside_dev
            
            # Factor metrics
            factor_metrics = {}
            if factor_loadings is not None and factor_returns is not None:
                # Factor exposures
                factor_exposures = factor_loadings.T @ weights
                factor_metrics['factor_exposures'] = factor_exposures.to_dict()
                
                # Factor risk decomposition
                factor_cov = factor_returns.cov()
                factor_risk = np.sqrt(factor_exposures @ factor_cov @ factor_exposures)
                factor_metrics['factor_risk'] = factor_risk
                
                # Factor contributions
                factor_contributions = factor_exposures * (factor_returns.mean() * 252)
                factor_metrics['factor_contributions'] = factor_contributions.to_dict()
            
            # Sector metrics
            sector_metrics = {}
            if sector_exposures is not None:
                # Sector exposures
                sector_metrics['sector_exposures'] = sector_exposures.to_dict()
                
                # Sector concentration
                sector_metrics['sector_concentration'] = (sector_exposures ** 2).sum()
            
            # Liquidity metrics
            turnover = self._calculate_turnover(weights)
            liquidity_score = self._calculate_liquidity_score(returns, weights)
            
            # Diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(returns, weights)
            
            # Benchmark metrics
            benchmark_metrics = {}
            if benchmark is not None:
                excess_return = portfolio_returns - benchmark
                tracking_error = excess_return.std() * np.sqrt(252)
                information_ratio = excess_return.mean() * np.sqrt(252) / tracking_error
                beta = portfolio_returns.cov(benchmark) / benchmark.var()
                alpha = annual_return - (
                    benchmark.mean() * 252 +
                    beta * (benchmark.mean() * 252)
                )
                
                benchmark_metrics = {
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha
                }
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'sortino_ratio': sortino_ratio,
                'turnover': turnover,
                'liquidity_score': liquidity_score,
                'diversification_ratio': diversification_ratio,
                **factor_metrics,
                **sector_metrics,
                **benchmark_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def mean_variance_optimize(
        self,
        returns: pd.DataFrame,
        target_volatility: Optional[float] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Mean-variance optimization with constraints.
        
        Args:
            returns: Asset returns
            target_volatility: Target portfolio volatility
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Optimal portfolio weights
        """
        try:
            # Calculate expected returns and covariance
            mu = returns.mean() * 252
            sigma = returns.cov() * 252
            
            # Define variables
            w = cp.Variable(len(returns.columns))
            
            # Define objective
            if target_volatility is not None:
                # Maximize return subject to volatility constraint
                objective = cp.Maximize(mu @ w)
                constraints = [
                    cp.quad_form(w, sigma) <= target_volatility ** 2
                ]
            else:
                # Maximize Sharpe ratio
                objective = cp.Maximize(mu @ w / cp.sqrt(cp.quad_form(w, sigma)))
                constraints = []
            
            # Add constraints
            constraints.extend([
                cp.sum(w) == 1,  # Full investment
                w >= self.constraints.min_weight,  # Minimum weight
                w <= self.constraints.max_weight,  # Maximum weight
                cp.sum(cp.abs(w)) <= self.constraints.max_leverage  # Leverage limit
            ])
            
            # Add factor constraints
            if factor_loadings is not None:
                for factor in factor_loadings.columns:
                    factor_exposure = factor_loadings[factor] @ w
                    constraints.extend([
                        factor_exposure >= self.constraints.min_factor_exposure,
                        factor_exposure <= self.constraints.max_factor_exposure
                    ])
            
            # Add sector constraints
            if sector_exposures is not None:
                for sector in sector_exposures.index:
                    sector_exposure = sector_exposures[sector] @ w
                    constraints.append(
                        sector_exposure <= self.constraints.max_sector_exposure
                    )
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status != 'optimal':
                raise ValueError(f"Optimization failed: {problem.status}")
            
            # Get optimal weights
            weights = pd.Series(w.value, index=returns.columns)
            
            # Store weights
            self.weights_history['mean_variance'] = weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            raise
    
    def risk_parity_optimize(
        self,
        returns: pd.DataFrame,
        target_risk: Optional[float] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Risk parity optimization with constraints.
        
        Args:
            returns: Asset returns
            target_risk: Target portfolio risk
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Optimal portfolio weights
        """
        try:
            # Calculate covariance
            sigma = returns.cov() * 252
            
            # Define variables
            w = cp.Variable(len(returns.columns))
            
            # Define objective
            if target_risk is not None:
                # Minimize deviation from equal risk contribution
                risk_contrib = cp.multiply(w, sigma @ w)
                objective = cp.Minimize(
                    cp.sum_squares(risk_contrib - cp.sum(risk_contrib) / len(w))
                )
                constraints = [
                    cp.sqrt(cp.quad_form(w, sigma)) == target_risk
                ]
            else:
                # Maximize diversification ratio
                vol = cp.sqrt(cp.diag(sigma))
                objective = cp.Maximize(
                    (vol @ w) / cp.sqrt(cp.quad_form(w, sigma))
                )
                constraints = []
            
            # Add constraints
            constraints.extend([
                cp.sum(w) == 1,  # Full investment
                w >= self.constraints.min_weight,  # Minimum weight
                w <= self.constraints.max_weight,  # Maximum weight
                cp.sum(cp.abs(w)) <= self.constraints.max_leverage  # Leverage limit
            ])
            
            # Add factor constraints
            if factor_loadings is not None:
                for factor in factor_loadings.columns:
                    factor_exposure = factor_loadings[factor] @ w
                    constraints.extend([
                        factor_exposure >= self.constraints.min_factor_exposure,
                        factor_exposure <= self.constraints.max_factor_exposure
                    ])
            
            # Add sector constraints
            if sector_exposures is not None:
                for sector in sector_exposures.index:
                    sector_exposure = sector_exposures[sector] @ w
                    constraints.append(
                        sector_exposure <= self.constraints.max_sector_exposure
                    )
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status != 'optimal':
                raise ValueError(f"Optimization failed: {problem.status}")
            
            # Get optimal weights
            weights = pd.Series(w.value, index=returns.columns)
            
            # Store weights
            self.weights_history['risk_parity'] = weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            raise
    
    def cvar_optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        CVaR optimization with constraints.
        
        Args:
            returns: Asset returns
            target_return: Target portfolio return
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Optimal portfolio weights
        """
        try:
            # Generate scenarios
            np.random.seed(self.random_state)
            scenarios = np.random.multivariate_normal(
                returns.mean(),
                returns.cov(),
                self.n_simulations
            )
            
            # Define variables
            w = cp.Variable(len(returns.columns))
            alpha = cp.Variable()  # VaR
            beta = cp.Variable(self.n_simulations)  # CVaR auxiliary variables
            
            # Define objective
            if target_return is not None:
                # Minimize CVaR subject to return constraint
                objective = cp.Minimize(
                    alpha + 1 / (0.05 * self.n_simulations) * cp.sum(beta)
                )
                constraints = [
                    returns.mean() @ w * 252 >= target_return
                ]
            else:
                # Maximize return/CVaR ratio
                objective = cp.Maximize(
                    (returns.mean() @ w * 252) /
                    (alpha + 1 / (0.05 * self.n_simulations) * cp.sum(beta))
                )
                constraints = []
            
            # Add CVaR constraints
            for i in range(self.n_simulations):
                constraints.append(
                    beta[i] >= -scenarios[i] @ w - alpha
                )
            
            # Add constraints
            constraints.extend([
                cp.sum(w) == 1,  # Full investment
                w >= self.constraints.min_weight,  # Minimum weight
                w <= self.constraints.max_weight,  # Maximum weight
                cp.sum(cp.abs(w)) <= self.constraints.max_leverage  # Leverage limit
            ])
            
            # Add factor constraints
            if factor_loadings is not None:
                for factor in factor_loadings.columns:
                    factor_exposure = factor_loadings[factor] @ w
                    constraints.extend([
                        factor_exposure >= self.constraints.min_factor_exposure,
                        factor_exposure <= self.constraints.max_factor_exposure
                    ])
            
            # Add sector constraints
            if sector_exposures is not None:
                for sector in sector_exposures.index:
                    sector_exposure = sector_exposures[sector] @ w
                    constraints.append(
                        sector_exposure <= self.constraints.max_sector_exposure
                    )
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status != 'optimal':
                raise ValueError(f"Optimization failed: {problem.status}")
            
            # Get optimal weights
            weights = pd.Series(w.value, index=returns.columns)
            
            # Store weights
            self.weights_history['cvar'] = weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in CVaR optimization: {str(e)}")
            raise
    
    def black_litterman_optimize(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Black-Litterman optimization with constraints.
        
        Args:
            returns: Asset returns
            market_caps: Market capitalizations
            views: Dictionary of return views
            view_confidences: Dictionary of view confidences
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Optimal portfolio weights
        """
        try:
            # Calculate market equilibrium
            market_weights = market_caps / market_caps.sum()
            sigma = returns.cov() * 252
            pi = returns.mean() * 252  # Equilibrium returns
            
            # Define variables
            w = cp.Variable(len(returns.columns))
            
            # Define objective
            if views is not None:
                # Incorporate views
                omega = np.diag([1 / view_confidences.get(k, 1) for k in views.keys()])
                P = np.zeros((len(views), len(returns.columns)))
                q = np.array(list(views.values()))
                
                for i, asset in enumerate(views.keys()):
                    P[i, returns.columns.get_loc(asset)] = 1
                
                # Black-Litterman expected returns
                tau = 0.05  # Prior uncertainty
                bl_returns = np.linalg.inv(
                    np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(omega) @ P
                ) @ (
                    np.linalg.inv(tau * sigma) @ pi + P.T @ np.linalg.inv(omega) @ q
                )
                
                # Maximize utility
                objective = cp.Maximize(
                    bl_returns @ w - 0.5 * cp.quad_form(w, sigma)
                )
            else:
                # Maximize utility with equilibrium returns
                objective = cp.Maximize(
                    pi @ w - 0.5 * cp.quad_form(w, sigma)
                )
            
            # Add constraints
            constraints = [
                cp.sum(w) == 1,  # Full investment
                w >= self.constraints.min_weight,  # Minimum weight
                w <= self.constraints.max_weight,  # Maximum weight
                cp.sum(cp.abs(w)) <= self.constraints.max_leverage  # Leverage limit
            ]
            
            # Add factor constraints
            if factor_loadings is not None:
                for factor in factor_loadings.columns:
                    factor_exposure = factor_loadings[factor] @ w
                    constraints.extend([
                        factor_exposure >= self.constraints.min_factor_exposure,
                        factor_exposure <= self.constraints.max_factor_exposure
                    ])
            
            # Add sector constraints
            if sector_exposures is not None:
                for sector in sector_exposures.index:
                    sector_exposure = sector_exposures[sector] @ w
                    constraints.append(
                        sector_exposure <= self.constraints.max_sector_exposure
                    )
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status != 'optimal':
                raise ValueError(f"Optimization failed: {problem.status}")
            
            # Get optimal weights
            weights = pd.Series(w.value, index=returns.columns)
            
            # Store weights
            self.weights_history['black_litterman'] = weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            raise
    
    def _calculate_turnover(self, weights: pd.Series) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Turnover ratio
        """
        try:
            if not self.weights_history:
                return 0
            
            # Calculate turnover from last weights
            last_weights = list(self.weights_history.values())[-1]
            turnover = np.abs(weights - last_weights).sum() / 2
            
            return turnover
            
        except Exception as e:
            logger.error(f"Error calculating turnover: {str(e)}")
            return 0
    
    def _calculate_liquidity_score(
        self,
        returns: pd.DataFrame,
        weights: pd.Series
    ) -> float:
        """
        Calculate portfolio liquidity score.
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
        
        Returns:
            Liquidity score
        """
        try:
            # Calculate average daily volume
            volumes = returns.abs().mean()
            
            # Calculate portfolio liquidity
            portfolio_liquidity = (volumes * weights).sum()
            
            return portfolio_liquidity
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0
    
    def _calculate_diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: pd.Series
    ) -> float:
        """
        Calculate portfolio diversification ratio.
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
        
        Returns:
            Diversification ratio
        """
        try:
            # Calculate individual volatilities
            vols = returns.std() * np.sqrt(252)
            
            # Calculate portfolio volatility
            sigma = returns.cov() * 252
            port_vol = np.sqrt(weights @ sigma @ weights)
            
            # Calculate weighted average volatility
            avg_vol = (vols * weights).sum()
            
            # Calculate diversification ratio
            div_ratio = avg_vol / port_vol
            
            return div_ratio
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 0
    
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
            metrics = self.calculate_metrics(returns, weights)
            
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
