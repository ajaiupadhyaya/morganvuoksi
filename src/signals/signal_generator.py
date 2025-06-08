from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from src.config import config
from dataclasses import dataclass
from scipy.stats import spearmanr
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Signal generation configuration."""
    min_probability: float = 0.6
    max_position_size: float = 0.2
    min_holding_period: int = 5
    max_holding_period: int = 20
    volatility_threshold: float = 0.3
    correlation_threshold: float = 0.7
    min_liquidity: float = 1e6
    max_drawdown: float = 0.15
    min_factor_exposure: float = 0.1
    max_factor_exposure: float = 0.5
    n_splits: int = 5
    random_state: int = 42

class SignalGenerationError(Exception):
    """Custom exception for signal generation errors."""
    pass

class SignalGenerator:
    """Advanced signal generation with risk management."""
    
    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        lookback_period: int = 252,
        n_models: int = 4
    ):
        """
        Initialize signal generator.
        
        Args:
            config: Signal generation configuration
            lookback_period: Lookback period for calculations
            n_models: Number of models to train
        """
        self.config = config or SignalConfig()
        self.lookback_period = lookback_period
        self.n_models = n_models
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            )
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize tracking
        self.feature_importance = {}
        self.model_performance = {}
        self.signal_history = {}
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for signal generation.
        
        Args:
            data: Price and volume data
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Features and target variables
        """
        try:
            # Calculate returns
            returns = data.pivot(columns='symbol', values='returns')
            
            # Calculate target variable (future returns)
            future_returns = returns.shift(-1)
            target = (future_returns > 0).astype(int)
            
            # Price-based features
            price_features = pd.DataFrame()
            for symbol in returns.columns:
                # Returns
                price_features[f'{symbol}_return'] = returns[symbol]
                price_features[f'{symbol}_return_5d'] = returns[symbol].rolling(5).mean()
                price_features[f'{symbol}_return_20d'] = returns[symbol].rolling(20).mean()
                
                # Volatility
                price_features[f'{symbol}_vol'] = returns[symbol].rolling(20).std()
                price_features[f'{symbol}_vol_ratio'] = (
                    returns[symbol].rolling(5).std() /
                    returns[symbol].rolling(20).std()
                )
                
                # Momentum
                price_features[f'{symbol}_momentum'] = (
                    returns[symbol].rolling(20).mean() /
                    returns[symbol].rolling(60).mean()
                )
                
                # Mean reversion
                price_features[f'{symbol}_mean_rev'] = (
                    returns[symbol].rolling(5).mean() /
                    returns[symbol].rolling(20).mean()
                )
            
            # Volume-based features
            volume_features = pd.DataFrame()
            for symbol in returns.columns:
                # Volume
                volume = data[data['symbol'] == symbol]['volume']
                volume_features[f'{symbol}_volume'] = volume
                volume_features[f'{symbol}_volume_ma'] = volume.rolling(20).mean()
                volume_features[f'{symbol}_volume_ratio'] = (
                    volume / volume.rolling(20).mean()
                )
                
                # Volume volatility
                volume_features[f'{symbol}_volume_vol'] = volume.rolling(20).std()
                volume_features[f'{symbol}_volume_vol_ratio'] = (
                    volume.rolling(5).std() /
                    volume.rolling(20).std()
                )
            
            # Factor-based features
            factor_features = pd.DataFrame()
            if factor_loadings is not None and factor_returns is not None:
                for factor in factor_loadings.columns:
                    # Factor returns
                    factor_features[f'{factor}_return'] = factor_returns[factor]
                    factor_features[f'{factor}_return_5d'] = (
                        factor_returns[factor].rolling(5).mean()
                    )
                    factor_features[f'{factor}_return_20d'] = (
                        factor_returns[factor].rolling(20).mean()
                    )
                    
                    # Factor volatility
                    factor_features[f'{factor}_vol'] = (
                        factor_returns[factor].rolling(20).std()
                    )
                    factor_features[f'{factor}_vol_ratio'] = (
                        factor_returns[factor].rolling(5).std() /
                        factor_returns[factor].rolling(20).std()
                    )
            
            # Market features
            market_features = pd.DataFrame()
            market_returns = returns.mean(axis=1)
            market_features['market_return'] = market_returns
            market_features['market_return_5d'] = market_returns.rolling(5).mean()
            market_features['market_return_20d'] = market_returns.rolling(20).mean()
            market_features['market_vol'] = market_returns.rolling(20).std()
            market_features['market_vol_ratio'] = (
                market_returns.rolling(5).std() /
                market_returns.rolling(20).std()
            )
            
            # Combine features
            features = pd.concat([
                price_features,
                volume_features,
                factor_features,
                market_features
            ], axis=1)
            
            # Drop NaN values
            features = features.dropna()
            target = target.loc[features.index]
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train models for signal generation.
        
        Args:
            features: Feature matrix
            target: Target variable
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Dictionary of model performance metrics
        """
        try:
            # Initialize cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            
            # Train models
            for name, model in self.models.items():
                # Initialize metrics
                metrics = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': []
                }
                
                # Cross-validation
                for train_idx, test_idx in tscv.split(features):
                    # Split data
                    X_train = features.iloc[train_idx]
                    y_train = target.iloc[train_idx]
                    X_test = features.iloc[test_idx]
                    y_test = target.iloc[test_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                    metrics['precision'].append(precision_score(y_test, y_pred))
                    metrics['recall'].append(recall_score(y_test, y_pred))
                    metrics['f1'].append(f1_score(y_test, y_pred))
                
                # Store average metrics
                self.model_performance[name] = {
                    metric: np.mean(values)
                    for metric, values in metrics.items()
                }
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = pd.Series(
                        model.feature_importances_,
                        index=features.columns
                    ).sort_values(ascending=False)
            
            return self.model_performance
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            features: Feature matrix
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            DataFrame of trading signals
        """
        try:
            # Initialize signals
            signals = pd.DataFrame(index=features.index)
            
            # Generate predictions
            for name, model in self.models.items():
                # Make predictions
                proba = model.predict_proba(features)
                signals[f'{name}_proba'] = proba[:, 1]
            
            # Calculate ensemble probability
            signals['ensemble_proba'] = signals.filter(like='_proba').mean(axis=1)
            
            # Generate signals
            signals['signal'] = 0
            signals.loc[signals['ensemble_proba'] > self.config.min_probability, 'signal'] = 1
            signals.loc[signals['ensemble_proba'] < (1 - self.config.min_probability), 'signal'] = -1
            
            # Apply risk filters
            if factor_loadings is not None and factor_returns is not None:
                # Factor exposure filter
                for factor in factor_loadings.columns:
                    factor_exposure = factor_loadings[factor]
                    factor_vol = factor_returns[factor].std()
                    
                    # Reduce position size for high factor exposure
                    mask = abs(factor_exposure) > self.config.max_factor_exposure
                    signals.loc[mask, 'signal'] *= 0.5
                    
                    # Reduce position size for high factor volatility
                    mask = factor_vol > self.config.volatility_threshold
                    signals.loc[mask, 'signal'] *= 0.5
            
            if sector_exposures is not None:
                # Sector concentration filter
                for sector in sector_exposures.index:
                    sector_exposure = sector_exposures[sector]
                    
                    # Reduce position size for high sector exposure
                    mask = abs(sector_exposure) > self.config.max_sector_exposure
                    signals.loc[mask, 'signal'] *= 0.5
            
            # Store signals
            self.signal_history = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def calculate_signal_metrics(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_exposures: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate signal performance metrics.
        
        Args:
            signals: Trading signals
            returns: Asset returns
            benchmark: Benchmark returns
            factor_loadings: Factor loadings
            factor_returns: Factor returns
            sector_exposures: Sector exposures
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (signals['signal'] * returns).mean(axis=1)
            
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
                factor_exposures = factor_loadings.T @ signals['signal']
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
            
            # Signal metrics
            signal_metrics = {
                'signal_accuracy': (
                    (signals['signal'] * returns).mean(axis=1) > 0
                ).mean(),
                'signal_precision': (
                    (signals['signal'] * returns).mean(axis=1) > 0
                ).mean(),
                'signal_recall': (
                    (signals['signal'] * returns).mean(axis=1) > 0
                ).mean(),
                'signal_f1': (
                    (signals['signal'] * returns).mean(axis=1) > 0
                ).mean()
            }
            
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
                **factor_metrics,
                **sector_metrics,
                **signal_metrics,
                **benchmark_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal metrics: {str(e)}")
            return {}

def main():
    """Main execution function."""
    try:
        # Initialize signal generator
        generator = SignalGenerator()
        
        # Load data
        data_path = config.paths.processed_data_dir / "processed_data_20240315_1200.parquet"
        df = pd.read_parquet(data_path)
        
        # Prepare features
        features, target = generator.prepare_features(df)
        
        # Train models
        model_performance = generator.train_models(features, target)
        
        # Generate signals
        signals = generator.generate_signals(features)
        
        # Calculate metrics
        metrics = generator.calculate_signal_metrics(signals, df)
        
        # Save results
        signals.to_parquet(
            config.paths.processed_data_dir / "trading_signals.parquet"
        )
        
        pd.DataFrame({
            'model_performance': model_performance,
            'signal_metrics': metrics
        }).to_parquet(
            config.paths.processed_data_dir / "signal_metrics.parquet"
        )
        
        logger.info("Signal generation completed successfully")
        
    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()