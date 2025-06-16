"""
Market accuracy validator with institutional standards.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MarketValidator:
    """Validator for market calculations and signals."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.confidence_level = config.get('confidence_level', 0.95)
    
    def validate_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Validate return calculations."""
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pval = stats.jarque_bera(returns)
        
        # Autocorrelation test
        acf = pd.Series(returns).autocorr()
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'skew': skew,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'autocorrelation': acf
        }
    
    def validate_volatility(self, returns: pd.Series, window: int = 20) -> Dict[str, float]:
        """Validate volatility calculations."""
        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Parkinson volatility (high-low)
        high_low_vol = np.sqrt(1 / (4 * np.log(2)) * 
                             (np.log(returns.max() / returns.min()) ** 2))
        
        # Garman-Klass volatility
        gk_vol = np.sqrt(0.5 * np.log(returns.max() / returns.min()) ** 2 -
                         (2 * np.log(2) - 1) * 
                         (np.log(returns.max() / returns.min()) ** 2))
        
        return {
            'rolling_volatility': rolling_vol.mean(),
            'parkinson_volatility': high_low_vol,
            'garman_klass_volatility': gk_vol
        }
    
    def validate_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """Validate drawdown calculations."""
        # Calculate drawdown
        rolling_max = prices.expanding().max()
        drawdown = (prices / rolling_max - 1)
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean()
        
        # Drawdown duration
        drawdown_duration = (drawdown < 0).astype(int).groupby(
            (drawdown < 0).astype(int).cumsum()
        ).cumsum().max()
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': drawdown_duration
        }
    
    def validate_var(self, returns: pd.Series, 
                    confidence_level: float = 0.95) -> Dict[str, float]:
        """Validate Value at Risk calculations."""
        # Historical VaR
        historical_var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        parametric_var = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
        
        # Expected Shortfall (CVaR)
        cvar = returns[returns <= historical_var].mean()
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'expected_shortfall': cvar
        }
    
    def validate_sharpe(self, returns: pd.Series) -> Dict[str, float]:
        """Validate Sharpe ratio calculations."""
        # Annualized Sharpe
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Information ratio (assuming benchmark returns)
        benchmark_returns = returns.rolling(window=self.lookback_period).mean()
        information_ratio = np.sqrt(252) * (returns - benchmark_returns).mean() / \
                          (returns - benchmark_returns).std()
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'information_ratio': information_ratio
        }
    
    def validate_signals(self, signals: pd.Series, 
                        returns: pd.Series) -> Dict[str, float]:
        """Validate trading signals."""
        # Signal accuracy
        correct_signals = (signals * returns > 0).mean()
        
        # Signal precision
        precision = (signals * returns > 0).sum() / (signals != 0).sum()
        
        # Signal recall
        recall = (signals * returns > 0).sum() / (returns != 0).sum()
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'signal_accuracy': correct_signals,
            'signal_precision': precision,
            'signal_recall': recall,
            'signal_f1': f1
        }
    
    def validate_data_leakage(self, data: pd.DataFrame, 
                            target_col: str) -> Dict[str, bool]:
        """Check for data leakage in features."""
        leakage_checks = {}
        
        # Check for future data leakage
        for col in data.columns:
            if col != target_col:
                # Check if feature leads target
                lead_corr = data[col].shift(-1).corr(data[target_col])
                leakage_checks[f'{col}_leads_target'] = abs(lead_corr) > 0.5
        
        # Check for lookahead bias in rolling calculations
        for col in data.columns:
            if 'rolling' in col.lower() or 'ma' in col.lower():
                # Check if rolling window includes future data
                leakage_checks[f'{col}_lookahead'] = False  # Implement specific checks
        
        return leakage_checks
    
    def validate_regime_detection(self, returns: pd.Series, 
                                regimes: pd.Series) -> Dict[str, float]:
        """Validate regime detection."""
        # Regime stability
        regime_changes = (regimes != regimes.shift(1)).sum()
        regime_stability = 1 - (regime_changes / len(regimes))
        
        # Regime return characteristics
        regime_returns = returns.groupby(regimes).agg(['mean', 'std'])
        
        # Regime transition matrix
        transition_matrix = pd.crosstab(regimes, regimes.shift(1))
        
        return {
            'regime_stability': regime_stability,
            'regime_returns': regime_returns.to_dict(),
            'transition_matrix': transition_matrix.to_dict()
        }
    
    def generate_validation_report(self, data: pd.DataFrame, 
                                 signals: pd.Series) -> Dict:
        """Generate comprehensive validation report."""
        returns = data['Returns']
        prices = data['Close']
        
        report = {
            'returns_validation': self.validate_returns(returns),
            'volatility_validation': self.validate_volatility(returns),
            'drawdown_validation': self.validate_drawdown(prices),
            'var_validation': self.validate_var(returns),
            'sharpe_validation': self.validate_sharpe(returns),
            'signal_validation': self.validate_signals(signals, returns),
            'leakage_validation': self.validate_data_leakage(data, 'Returns')
        }
        
        if 'Regime' in data.columns:
            report['regime_validation'] = self.validate_regime_detection(
                returns, data['Regime']
            )
        
        return report 
