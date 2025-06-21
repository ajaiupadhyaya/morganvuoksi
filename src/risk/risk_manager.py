"""
Risk Management Module
Comprehensive risk management system with VaR, CVaR, stress testing, and position sizing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    var_confidence_level: float = 0.95
    max_position_size: float = 0.1  # 10% max position
    max_portfolio_risk: float = 0.02  # 2% max portfolio VaR
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    correlation_threshold: float = 0.7  # Maximum correlation between positions
    volatility_lookback: int = 252  # Days for volatility calculation
    stress_test_scenarios: Dict = None

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # Initialize stress test scenarios
        if self.config.stress_test_scenarios is None:
            self.config.stress_test_scenarios = {
                'market_crash': -0.20,
                'recession': -0.10,
                'volatility_spike': 0.50,
                'interest_rate_hike': 0.02,
                'currency_crisis': -0.15,
                'commodity_shock': -0.25
            }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = None, 
                     method: str = "historical") -> Dict:
        """Calculate Value at Risk (VaR)."""
        
        if confidence_level is None:
            confidence_level = self.config.var_confidence_level
        
        if method == "historical":
            return self._historical_var(returns, confidence_level)
        elif method == "parametric":
            return self._parametric_var(returns, confidence_level)
        elif method == "monte_carlo":
            return self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Calculate historical VaR."""
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return {
            'var': var,
            'confidence_level': confidence_level,
            'method': 'historical',
            'lookback_period': len(returns)
        }
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> Dict:
        """Calculate parametric VaR assuming normal distribution."""
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)
        
        var = mean_return - z_score * std_return
        
        return {
            'var': var,
            'confidence_level': confidence_level,
            'method': 'parametric',
            'mean': mean_return,
            'std': std_return,
            'z_score': z_score
        }
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                        num_simulations: int = 10000) -> Dict:
        """Calculate VaR using Monte Carlo simulation."""
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return {
            'var': var,
            'confidence_level': confidence_level,
            'method': 'monte_carlo',
            'num_simulations': num_simulations,
            'mean': mean_return,
            'std': std_return
        }
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = None) -> Dict:
        """Calculate Conditional Value at Risk (CVaR)."""
        
        if confidence_level is None:
            confidence_level = self.config.var_confidence_level
        
        var = self.calculate_var(returns, confidence_level)['var']
        cvar = returns[returns <= var].mean()
        
        return {
            'cvar': cvar,
            'var': var,
            'confidence_level': confidence_level,
            'tail_loss': cvar - var
        }
    
    def calculate_portfolio_risk(self, returns: pd.DataFrame, 
                               weights: pd.Series) -> Dict:
        """Calculate comprehensive portfolio risk metrics."""
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        var_95 = self.calculate_var(portfolio_returns, 0.95)['var']
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)['cvar']
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Component VaR
        component_var = self._calculate_component_var(returns, weights, var_95)
        
        # Marginal VaR
        marginal_var = self._calculate_marginal_var(returns, weights, var_95)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(weights)
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(returns, weights)
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'component_var': component_var,
            'marginal_var': marginal_var,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'portfolio_returns': portfolio_returns
        }
    
    def _calculate_component_var(self, returns: pd.DataFrame, 
                               weights: pd.Series, portfolio_var: float) -> pd.Series:
        """Calculate Component VaR for each asset."""
        
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Marginal contribution to risk
        marginal_contribution = np.dot(cov_matrix, weights) / portfolio_vol
        
        # Component VaR
        component_var = weights * marginal_contribution * portfolio_var / portfolio_vol
        
        return pd.Series(component_var, index=returns.columns)
    
    def _calculate_marginal_var(self, returns: pd.DataFrame, 
                              weights: pd.Series, portfolio_var: float) -> pd.Series:
        """Calculate Marginal VaR for each asset."""
        
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Marginal VaR
        marginal_var = np.dot(cov_matrix, weights) * portfolio_var / portfolio_vol
        
        return pd.Series(marginal_var, index=returns.columns)
    
    def _calculate_concentration_risk(self, weights: pd.Series) -> float:
        """Calculate concentration risk using Herfindahl index."""
        
        herfindahl_index = np.sum(weights ** 2)
        return herfindahl_index
    
    def _calculate_correlation_risk(self, returns: pd.DataFrame, 
                                  weights: pd.Series) -> float:
        """Calculate correlation risk."""
        
        corr_matrix = returns.corr()
        
        # Weighted average correlation
        weighted_corr = 0
        total_weight = 0
        
        for i, asset1 in enumerate(returns.columns):
            for j, asset2 in enumerate(returns.columns):
                if i != j:
                    weight_product = weights[asset1] * weights[asset2]
                    weighted_corr += weight_product * corr_matrix.iloc[i, j]
                    total_weight += weight_product
        
        if total_weight > 0:
            avg_correlation = weighted_corr / total_weight
        else:
            avg_correlation = 0
        
        return avg_correlation
    
    def run_stress_tests(self, returns: pd.DataFrame, 
                        weights: pd.Series) -> Dict:
        """Run comprehensive stress tests."""
        
        stress_results = {}
        
        for scenario_name, impact in self.config.stress_test_scenarios.items():
            stress_results[scenario_name] = self._run_stress_scenario(
                returns, weights, scenario_name, impact
            )
        
        return stress_results
    
    def _run_stress_scenario(self, returns: pd.DataFrame, 
                           weights: pd.Series, 
                           scenario_name: str, 
                           impact: float) -> Dict:
        """Run individual stress test scenario."""
        
        # Apply stress impact to returns
        stressed_returns = returns.copy()
        
        if scenario_name == "market_crash":
            # Market-wide decline
            stressed_returns = returns * (1 + impact)
        elif scenario_name == "volatility_spike":
            # Increase volatility
            stressed_returns = returns * (1 + np.random.normal(0, impact, len(returns)))
        elif scenario_name == "interest_rate_hike":
            # Interest rate impact (affects bond-like assets more)
            # This is a simplified model
            stressed_returns = returns * (1 + impact * 0.5)
        else:
            # Generic stress
            stressed_returns = returns * (1 + impact)
        
        # Calculate stressed portfolio metrics
        stressed_portfolio_returns = (stressed_returns * weights).sum(axis=1)
        stressed_var = self.calculate_var(stressed_portfolio_returns, 0.95)['var']
        stressed_cvar = self.calculate_cvar(stressed_portfolio_returns, 0.95)['cvar']
        
        # Calculate impact
        base_portfolio_returns = (returns * weights).sum(axis=1)
        base_var = self.calculate_var(base_portfolio_returns, 0.95)['var']
        base_cvar = self.calculate_cvar(base_portfolio_returns, 0.95)['cvar']
        
        var_impact = (stressed_var - base_var) / abs(base_var) if base_var != 0 else 0
        cvar_impact = (stressed_cvar - base_cvar) / abs(base_cvar) if base_cvar != 0 else 0
        
        return {
            'scenario': scenario_name,
            'impact': impact,
            'stressed_var': stressed_var,
            'stressed_cvar': stressed_cvar,
            'var_impact': var_impact,
            'cvar_impact': cvar_impact,
            'portfolio_loss': stressed_portfolio_returns.sum()
        }
    
    def calculate_position_size(self, price: float, 
                              volatility: float, 
                              account_size: float,
                              risk_per_trade: float = 0.02) -> Dict:
        """Calculate optimal position size using Kelly Criterion and risk management."""
        
        # Kelly Criterion (simplified)
        win_rate = 0.55  # Placeholder - should be calculated from historical data
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Risk-based position sizing
        risk_amount = account_size * risk_per_trade
        stop_loss_amount = price * self.config.stop_loss_pct
        position_size_risk = risk_amount / stop_loss_amount
        
        # Volatility-based position sizing
        volatility_scale = 1 / (volatility * np.sqrt(252))  # Annualized volatility
        position_size_vol = account_size * 0.1 * volatility_scale  # 10% of account
        
        # Take the minimum of different approaches
        optimal_size = min(position_size_risk, position_size_vol, 
                          account_size * kelly_fraction)
        
        # Apply maximum position size limit
        max_size = account_size * self.config.max_position_size
        final_size = min(optimal_size, max_size)
        
        return {
            'position_size': final_size,
            'position_value': final_size * price,
            'risk_amount': risk_amount,
            'kelly_fraction': kelly_fraction,
            'risk_based_size': position_size_risk,
            'volatility_based_size': position_size_vol
        }
    
    def check_risk_limits(self, portfolio_risk: Dict, 
                         position_risks: Dict) -> Dict:
        """Check if portfolio and positions are within risk limits."""
        
        violations = []
        warnings = []
        
        # Portfolio VaR check
        if portfolio_risk['var_95'] > self.config.max_portfolio_risk:
            violations.append(f"Portfolio VaR ({portfolio_risk['var_95']:.4f}) exceeds limit ({self.config.max_portfolio_risk:.4f})")
        
        # Maximum drawdown check
        if portfolio_risk['max_drawdown'] < -self.config.max_drawdown_limit:
            violations.append(f"Maximum drawdown ({portfolio_risk['max_drawdown']:.2%}) exceeds limit ({self.config.max_drawdown_limit:.2%})")
        
        # Position size checks
        for asset, risk in position_risks.items():
            if risk['position_size'] > self.config.max_position_size:
                violations.append(f"Position size for {asset} ({risk['position_size']:.2%}) exceeds limit ({self.config.max_position_size:.2%})")
        
        # Correlation checks
        if portfolio_risk['correlation_risk'] > self.config.correlation_threshold:
            warnings.append(f"High portfolio correlation ({portfolio_risk['correlation_risk']:.2f})")
        
        # Concentration checks
        if portfolio_risk['concentration_risk'] > 0.5:  # Herfindahl index threshold
            warnings.append(f"High portfolio concentration ({portfolio_risk['concentration_risk']:.2f})")
        
        return {
            'violations': violations,
            'warnings': warnings,
            'within_limits': len(violations) == 0,
            'risk_score': len(violations) * 10 + len(warnings) * 2  # Simple risk scoring
        }
    
    def generate_risk_report(self, returns: pd.DataFrame, 
                           weights: pd.Series) -> Dict:
        """Generate comprehensive risk report."""
        
        # Calculate all risk metrics
        portfolio_risk = self.calculate_portfolio_risk(returns, weights)
        stress_tests = self.run_stress_tests(returns, weights)
        
        # Calculate position-level risks
        position_risks = {}
        for asset in returns.columns:
            asset_returns = returns[asset]
            asset_vol = asset_returns.std() * np.sqrt(252)
            position_risks[asset] = {
                'volatility': asset_vol,
                'var_95': self.calculate_var(asset_returns, 0.95)['var'],
                'position_size': weights[asset],
                'component_var': portfolio_risk['component_var'][asset],
                'marginal_var': portfolio_risk['marginal_var'][asset]
            }
        
        # Check risk limits
        limit_check = self.check_risk_limits(portfolio_risk, position_risks)
        
        return {
            'portfolio_risk': portfolio_risk,
            'position_risks': position_risks,
            'stress_tests': stress_tests,
            'limit_check': limit_check,
            'report_date': datetime.now(),
            'summary': {
                'total_risk': portfolio_risk['var_95'],
                'max_drawdown': portfolio_risk['max_drawdown'],
                'risk_score': limit_check['risk_score'],
                'violations': len(limit_check['violations']),
                'warnings': len(limit_check['warnings'])
            }
        } 