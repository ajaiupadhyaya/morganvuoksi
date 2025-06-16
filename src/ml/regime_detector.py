"""
Modular regime detection system for market state analysis.
Implements market breadth, volatility term structure, correlation, and liquidity indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeState(Enum):
    """Enum for regime states."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEUTRAL = "neutral"
    EXTREME = "extreme"

@dataclass
class RegimeIndicator:
    """Base class for regime indicators."""
    name: str
    state: RegimeState
    value: float
    threshold: float
    timestamp: datetime

class MarketBreadthIndicator:
    """Market breadth indicator using moving averages and advance-decline ratios."""
    
    def __init__(
        self,
        ma_short: int = 50,
        ma_long: int = 200,
        ad_ratio_window: int = 20
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.ad_ratio_window = ad_ratio_window
        self.history: List[RegimeIndicator] = []
    
    def calculate(
        self,
        prices: pd.DataFrame,
        advances: pd.Series,
        declines: pd.Series
    ) -> RegimeIndicator:
        """Calculate market breadth indicators."""
        try:
            # Calculate % of assets above moving averages
            above_ma_short = (prices > prices.rolling(self.ma_short).mean()).mean(axis=1)
            above_ma_long = (prices > prices.rolling(self.ma_long).mean()).mean(axis=1)
            
            # Calculate advance-decline ratio
            ad_ratio = (advances / declines).rolling(self.ad_ratio_window).mean()
            
            # Combine indicators
            breadth_score = (
                0.4 * above_ma_short.iloc[-1] +
                0.4 * above_ma_long.iloc[-1] +
                0.2 * ad_ratio.iloc[-1]
            )
            
            # Determine regime state
            if breadth_score > 0.7:
                state = RegimeState.HIGH
            elif breadth_score > 0.5:
                state = RegimeState.MEDIUM
            elif breadth_score > 0.3:
                state = RegimeState.LOW
            else:
                state = RegimeState.EXTREME
            
            indicator = RegimeIndicator(
                name="market_breadth",
                state=state,
                value=breadth_score,
                threshold=0.5,
                timestamp=datetime.now()
            )
            
            self.history.append(indicator)
            return indicator
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {str(e)}")
            return RegimeIndicator(
                name="market_breadth",
                state=RegimeState.NEUTRAL,
                value=0.5,
                threshold=0.5,
                timestamp=datetime.now()
            )

class VolatilityTermStructureIndicator:
    """Volatility term structure indicator using implied and realized volatility."""
    
    def __init__(
        self,
        short_term_window: int = 20,
        long_term_window: int = 60,
        spread_threshold: float = 0.1
    ):
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window
        self.spread_threshold = spread_threshold
        self.history: List[RegimeIndicator] = []
    
    def calculate(
        self,
        implied_vol: pd.DataFrame,
        realized_vol: pd.Series
    ) -> RegimeIndicator:
        """Calculate volatility term structure indicators."""
        try:
            # Calculate term structure slope
            short_term_iv = implied_vol.iloc[:, 0].rolling(self.short_term_window).mean()
            long_term_iv = implied_vol.iloc[:, -1].rolling(self.long_term_window).mean()
            term_structure_slope = (long_term_iv - short_term_iv) / short_term_iv
            
            # Calculate IV-RV spread
            iv_rv_spread = (
                implied_vol.iloc[:, 0].rolling(self.short_term_window).mean() -
                realized_vol.rolling(self.short_term_window).mean()
            ) / realized_vol.rolling(self.short_term_window).mean()
            
            # Combine indicators
            vol_score = (
                0.6 * term_structure_slope.iloc[-1] +
                0.4 * iv_rv_spread.iloc[-1]
            )
            
            # Determine regime state
            if vol_score > self.spread_threshold:
                state = RegimeState.HIGH
            elif vol_score > 0:
                state = RegimeState.MEDIUM
            elif vol_score > -self.spread_threshold:
                state = RegimeState.LOW
            else:
                state = RegimeState.EXTREME
            
            indicator = RegimeIndicator(
                name="volatility_term_structure",
                state=state,
                value=vol_score,
                threshold=self.spread_threshold,
                timestamp=datetime.now()
            )
            
            self.history.append(indicator)
            return indicator
            
        except Exception as e:
            logger.error(f"Error calculating volatility term structure: {str(e)}")
            return RegimeIndicator(
                name="volatility_term_structure",
                state=RegimeState.NEUTRAL,
                value=0.0,
                threshold=self.spread_threshold,
                timestamp=datetime.now()
            )

class CorrelationRegimeIndicator:
    """Correlation regime indicator using rolling correlation matrices."""
    
    def __init__(
        self,
        window: int = 60,
        spike_threshold: float = 0.8,
        dispersion_threshold: float = 0.3
    ):
        self.window = window
        self.spike_threshold = spike_threshold
        self.dispersion_threshold = dispersion_threshold
        self.history: List[RegimeIndicator] = []
    
    def calculate(
        self,
        returns: pd.DataFrame
    ) -> RegimeIndicator:
        """Calculate correlation regime indicators."""
        try:
            # Calculate rolling correlation matrix
            corr_matrix = returns.rolling(self.window).corr()
            
            # Calculate mean correlation
            mean_corr = corr_matrix.mean().mean()
            
            # Calculate correlation dispersion
            corr_dispersion = corr_matrix.std().mean()
            
            # Detect correlation spikes
            spike_count = (corr_matrix > self.spike_threshold).sum().sum()
            spike_ratio = spike_count / (len(returns.columns) ** 2)
            
            # Combine indicators
            corr_score = (
                0.4 * mean_corr +
                0.3 * corr_dispersion +
                0.3 * spike_ratio
            )
            
            # Determine regime state
            if corr_score > 0.7:
                state = RegimeState.HIGH
            elif corr_score > 0.5:
                state = RegimeState.MEDIUM
            elif corr_score > 0.3:
                state = RegimeState.LOW
            else:
                state = RegimeState.EXTREME
            
            indicator = RegimeIndicator(
                name="correlation_regime",
                state=state,
                value=corr_score,
                threshold=0.5,
                timestamp=datetime.now()
            )
            
            self.history.append(indicator)
            return indicator
            
        except Exception as e:
            logger.error(f"Error calculating correlation regime: {str(e)}")
            return RegimeIndicator(
                name="correlation_regime",
                state=RegimeState.NEUTRAL,
                value=0.5,
                threshold=0.5,
                timestamp=datetime.now()
            )

class LiquidityRegimeIndicator:
    """Liquidity regime indicator using volume and spread metrics."""
    
    def __init__(
        self,
        volume_window: int = 20,
        spread_window: int = 20,
        turnover_window: int = 20
    ):
        self.volume_window = volume_window
        self.spread_window = spread_window
        self.turnover_window = turnover_window
        self.history: List[RegimeIndicator] = []
    
    def calculate(
        self,
        volume: pd.DataFrame,
        spreads: pd.DataFrame,
        turnover: pd.DataFrame
    ) -> RegimeIndicator:
        """Calculate liquidity regime indicators."""
        try:
            # Calculate volume trends
            volume_ma = volume.rolling(self.volume_window).mean()
            volume_trend = (volume / volume_ma - 1).mean(axis=1)
            
            # Calculate spread trends
            spread_ma = spreads.rolling(self.spread_window).mean()
            spread_trend = (spreads / spread_ma - 1).mean(axis=1)
            
            # Calculate turnover trends
            turnover_ma = turnover.rolling(self.turnover_window).mean()
            turnover_trend = (turnover / turnover_ma - 1).mean(axis=1)
            
            # Combine indicators
            liquidity_score = (
                0.4 * volume_trend.iloc[-1] +
                0.3 * (-spread_trend.iloc[-1]) +  # Negative because higher spreads = lower liquidity
                0.3 * turnover_trend.iloc[-1]
            )
            
            # Determine regime state
            if liquidity_score > 0.1:
                state = RegimeState.HIGH
            elif liquidity_score > 0:
                state = RegimeState.MEDIUM
            elif liquidity_score > -0.1:
                state = RegimeState.LOW
            else:
                state = RegimeState.EXTREME
            
            indicator = RegimeIndicator(
                name="liquidity_regime",
                state=state,
                value=liquidity_score,
                threshold=0.0,
                timestamp=datetime.now()
            )
            
            self.history.append(indicator)
            return indicator
            
        except Exception as e:
            logger.error(f"Error calculating liquidity regime: {str(e)}")
            return RegimeIndicator(
                name="liquidity_regime",
                state=RegimeState.NEUTRAL,
                value=0.0,
                threshold=0.0,
                timestamp=datetime.now()
            )

class RegimeDetector:
    """Main regime detector combining multiple indicators."""
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        
        # Initialize indicators
        self.breadth_indicator = MarketBreadthIndicator(
            ma_short=self.config.get('ma_short', 50),
            ma_long=self.config.get('ma_long', 200),
            ad_ratio_window=self.config.get('ad_ratio_window', 20)
        )
        
        self.vol_indicator = VolatilityTermStructureIndicator(
            short_term_window=self.config.get('short_term_window', 20),
            long_term_window=self.config.get('long_term_window', 60),
            spread_threshold=self.config.get('spread_threshold', 0.1)
        )
        
        self.corr_indicator = CorrelationRegimeIndicator(
            window=self.config.get('corr_window', 60),
            spike_threshold=self.config.get('spike_threshold', 0.8),
            dispersion_threshold=self.config.get('dispersion_threshold', 0.3)
        )
        
        self.liq_indicator = LiquidityRegimeIndicator(
            volume_window=self.config.get('volume_window', 20),
            spread_window=self.config.get('spread_window', 20),
            turnover_window=self.config.get('turnover_window', 20)
        )
        
        self.regime_history: List[Dict] = []
    
    def detect_regime(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[str, float]]:
        """Detect current market regime using all indicators."""
        try:
            # Calculate individual indicators
            breadth = self.breadth_indicator.calculate(
                market_data['prices'],
                market_data['advances'],
                market_data['declines']
            )
            
            vol = self.vol_indicator.calculate(
                market_data['implied_vol'],
                market_data['realized_vol']
            )
            
            corr = self.corr_indicator.calculate(
                market_data['returns']
            )
            
            liq = self.liq_indicator.calculate(
                market_data['volume'],
                market_data['spreads'],
                market_data['turnover']
            )
            
            # Combine indicators into composite regime
            regime = {
                'timestamp': datetime.now(),
                'indicators': {
                    'breadth': {
                        'state': breadth.state.value,
                        'value': breadth.value
                    },
                    'volatility': {
                        'state': vol.state.value,
                        'value': vol.value
                    },
                    'correlation': {
                        'state': corr.state.value,
                        'value': corr.value
                    },
                    'liquidity': {
                        'state': liq.state.value,
                        'value': liq.value
                    }
                },
                'composite': self._calculate_composite_regime(
                    breadth, vol, corr, liq
                )
            }
            
            self.regime_history.append(regime)
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'indicators': {},
                'composite': 'unknown'
            }
    
    def _calculate_composite_regime(
        self,
        breadth: RegimeIndicator,
        vol: RegimeIndicator,
        corr: RegimeIndicator,
        liq: RegimeIndicator
    ) -> str:
        """Calculate composite regime label."""
        # Count extreme states
        extreme_count = sum(
            1 for ind in [breadth, vol, corr, liq]
            if ind.state == RegimeState.EXTREME
        )
        
        if extreme_count >= 2:
            return "extreme"
        
        # Count high states
        high_count = sum(
            1 for ind in [breadth, vol, corr, liq]
            if ind.state == RegimeState.HIGH
        )
        
        if high_count >= 2:
            return "high"
        
        # Count low states
        low_count = sum(
            1 for ind in [breadth, vol, corr, liq]
            if ind.state == RegimeState.LOW
        )
        
        if low_count >= 2:
            return "low"
        
        return "neutral"
    
    def get_regime_weights(self) -> Dict[str, float]:
        """Get model weights based on current regime."""
        try:
            if not self.regime_history:
                return {
                    'xgb': 0.33,
                    'lstm': 0.33,
                    'transformer': 0.34
                }
            
            current_regime = self.regime_history[-1]['composite']
            
            # Define regime-specific weights
            weights = {
                'extreme': {
                    'xgb': 0.2,
                    'lstm': 0.4,
                    'transformer': 0.4
                },
                'high': {
                    'xgb': 0.3,
                    'lstm': 0.4,
                    'transformer': 0.3
                },
                'low': {
                    'xgb': 0.4,
                    'lstm': 0.3,
                    'transformer': 0.3
                },
                'neutral': {
                    'xgb': 0.33,
                    'lstm': 0.33,
                    'transformer': 0.34
                }
            }
            
            return weights.get(current_regime, weights['neutral'])
            
        except Exception as e:
            logger.error(f"Error getting regime weights: {str(e)}")
            return {
                'xgb': 0.33,
                'lstm': 0.33,
                'transformer': 0.34
            }
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        
        history = []
        for regime in self.regime_history:
            row = {
                'timestamp': regime['timestamp'],
                'composite': regime['composite']
            }
            
            for indicator, data in regime['indicators'].items():
                row[f'{indicator}_state'] = data['state']
                row[f'{indicator}_value'] = data['value']
            
            history.append(row)
        
        return pd.DataFrame(history) 
