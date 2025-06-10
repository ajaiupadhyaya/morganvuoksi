"""
Quantitative research infrastructure for systematic trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ResearchInfrastructure:
    """Quantitative research infrastructure."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.factors = {}
        self.risk_models = {}
    
    def compute_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute factor exposures.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of factor exposures
        """
        try:
            # Size factor
            self.factors['size'] = np.log(data['market_cap'])
            
            # Value factor
            self.factors['value'] = data['book_value'] / data['market_cap']
            
            # Momentum factor
            self.factors['momentum'] = data['returns'].rolling(window=252).mean()
            
            # Volatility factor
            self.factors['volatility'] = data['returns'].rolling(window=252).std()
            
            # Quality factor
            self.factors['quality'] = data['roe'] * data['profit_margin']
            
            # Growth factor
            self.factors['growth'] = data['revenue_growth'] * data['earnings_growth']
            
            return self.factors
            
        except Exception as e:
            logger.error(f"Error computing factors: {str(e)}")
            return {}
    
    def compute_statistical_factors(self, data: pd.DataFrame,
                                  n_factors: int = 5) -> pd.DataFrame:
        """
        Compute statistical factors using PCA.
        
        Args:
            data: Market data
            n_factors: Number of factors to compute
            
        Returns:
            Statistical factors
        """
        try:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Compute PCA
            pca = PCA(n_components=n_factors)
            factors = pca.fit_transform(scaled_data)
            
            # Create factor DataFrame
            factor_df = pd.DataFrame(
                factors,
                columns=[f'factor_{i+1}' for i in range(n_factors)],
                index=data.index
            )
            
            # Store factor loadings
            self.factor_loadings = pd.DataFrame(
                pca.components_,
                columns=data.columns,
                index=[f'factor_{i+1}' for i in range(n_factors)]
            )
            
            return factor_df
            
        except Exception as e:
            logger.error(f"Error computing statistical factors: {str(e)}")
            return pd.DataFrame()
    
    def compute_risk_metrics(self, returns: pd.Series,
                           confidence_level: float = 0.95) -> Dict:
        """
        Compute risk metrics.
        
        Args:
            returns: Return series
            confidence_level: Confidence level for VaR/CVaR
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            # Value at Risk
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            # Conditional Value at Risk
            cvar = returns[returns <= var].mean()
            
            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / volatility
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (returns.mean() * 252 - risk_free_rate) / downside_volatility
            
            return {
                'var': var,
                'cvar': cvar,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            logger.error(f"Error computing risk metrics: {str(e)}")
            return {}
    
    def compute_regime_switching(self, returns: pd.Series,
                               n_regimes: int = 2) -> Dict:
        """
        Compute regime switching model.
        
        Args:
            returns: Return series
            n_regimes: Number of regimes
            
        Returns:
            Regime switching results
        """
        try:
            # Fit Markov switching model
            model = sm.tsa.MarkovRegression(
                returns,
                k_regimes=n_regimes,
                switching_variance=True
            )
            results = model.fit()
            
            # Get regime probabilities
            regime_probs = results.smoothed_marginal_probabilities
            
            # Get regime parameters
            regime_params = {
                'mean': results.regime_means,
                'variance': results.regime_variances,
                'transition': results.regime_transition
            }
            
            return {
                'probabilities': regime_probs,
                'parameters': regime_params
            }
            
        except Exception as e:
            logger.error(f"Error computing regime switching: {str(e)}")
            return {}
    
    def compute_cointegration(self, data: pd.DataFrame) -> Dict:
        """
        Compute cointegration relationships.
        
        Args:
            data: Price data
            
        Returns:
            Cointegration results
        """
        try:
            # Test for cointegration
            results = {}
            for col1 in data.columns:
                for col2 in data.columns:
                    if col1 < col2:
                        # Compute spread
                        spread = data[col1] - data[col2]
                        
                        # Test for stationarity
                        adf_result = adfuller(spread)
                        
                        if adf_result[1] < 0.05:  # Stationary at 5% level
                            results[f"{col1}-{col2}"] = {
                                'adf_statistic': adf_result[0],
                                'p_value': adf_result[1],
                                'spread': spread
                            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing cointegration: {str(e)}")
            return {}
    
    def compute_volatility_forecast(self, returns: pd.Series,
                                  model: str = 'garch') -> pd.Series:
        """
        Compute volatility forecast.
        
        Args:
            returns: Return series
            model: Volatility model to use
            
        Returns:
            Volatility forecast
        """
        try:
            if model == 'garch':
                # Fit GARCH model
                garch = arch_model(returns, vol='Garch', p=1, q=1)
                results = garch.fit(disp='off')
                
                # Get forecast
                forecast = results.forecast(horizon=1)
                volatility = np.sqrt(forecast.variance.values[-1, 0])
                
            elif model == 'ewma':
                # Compute EWMA volatility
                lambda_ = 0.94  # RiskMetrics lambda
                volatility = returns.ewm(alpha=1-lambda_).std().iloc[-1]
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error computing volatility forecast: {str(e)}")
            return pd.Series()
    
    def compute_correlation_forecast(self, returns: pd.DataFrame,
                                   method: str = 'ewma') -> pd.DataFrame:
        """
        Compute correlation forecast.
        
        Args:
            returns: Return data
            method: Correlation estimation method
            
        Returns:
            Correlation forecast
        """
        try:
            if method == 'ewma':
                # Compute EWMA correlation
                lambda_ = 0.94  # RiskMetrics lambda
                correlation = returns.ewm(alpha=1-lambda_).corr().iloc[-len(returns.columns):]
            
            elif method == 'dcc':
                # Implement DCC-GARCH
                pass
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error computing correlation forecast: {str(e)}")
            return pd.DataFrame()
    
    def compute_stress_test(self, portfolio: pd.Series,
                          scenarios: Dict[str, float]) -> Dict:
        """
        Compute stress test results.
        
        Args:
            portfolio: Portfolio weights
            scenarios: Dictionary of scenario returns
            
        Returns:
            Stress test results
        """
        try:
            results = {}
            
            for scenario, returns in scenarios.items():
                # Compute scenario return
                scenario_return = (portfolio * returns).sum()
                
                # Compute scenario risk
                scenario_risk = np.sqrt(
                    portfolio.dot(self.compute_correlation_forecast(returns))
                    .dot(portfolio)
                )
                
                results[scenario] = {
                    'return': scenario_return,
                    'risk': scenario_risk,
                    'sharpe': scenario_return / scenario_risk
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing stress test: {str(e)}")
            return {} 