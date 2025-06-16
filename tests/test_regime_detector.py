"""
Tests for the regime detection system.
Validates market breadth, volatility term structure, correlation, and liquidity indicators.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.ml.regime_detector import (
    RegimeState,
    RegimeIndicator,
    MarketBreadthIndicator,
    VolatilityTermStructureIndicator,
    CorrelationRegimeIndicator,
    LiquidityRegimeIndicator,
    RegimeDetector
)

class TestRegimeDetector(unittest.TestCase):
    """Test suite for regime detection system."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Generate test data
        np.random.seed(42)
        self.n_samples = 1000
        self.n_assets = 10
        
        # Generate price data
        self.prices = pd.DataFrame(
            np.random.lognormal(0, 0.1, (self.n_samples, self.n_assets)),
            columns=[f'asset_{i}' for i in range(self.n_assets)]
        )
        
        # Generate returns
        self.returns = self.prices.pct_change()
        
        # Generate advance-decline data
        self.advances = pd.Series(
            np.random.randint(100, 1000, self.n_samples),
            index=self.prices.index
        )
        self.declines = pd.Series(
            np.random.randint(100, 1000, self.n_samples),
            index=self.prices.index
        )
        
        # Generate volatility data
        self.implied_vol = pd.DataFrame(
            np.random.uniform(0.1, 0.3, (self.n_samples, 3)),
            columns=['1m', '3m', '6m']
        )
        self.realized_vol = pd.Series(
            np.random.uniform(0.1, 0.3, self.n_samples),
            index=self.prices.index
        )
        
        # Generate liquidity data
        self.volume = pd.DataFrame(
            np.random.lognormal(10, 1, (self.n_samples, self.n_assets)),
            columns=[f'asset_{i}' for i in range(self.n_assets)]
        )
        self.spreads = pd.DataFrame(
            np.random.uniform(0.001, 0.01, (self.n_samples, self.n_assets)),
            columns=[f'asset_{i}' for i in range(self.n_assets)]
        )
        self.turnover = pd.DataFrame(
            np.random.lognormal(12, 1, (self.n_samples, self.n_assets)),
            columns=[f'asset_{i}' for i in range(self.n_assets)]
        )
        
        # Initialize regime detector
        self.detector = RegimeDetector()
    
    def test_market_breadth(self):
        """Test market breadth indicator."""
        indicator = self.detector.breadth_indicator.calculate(
            self.prices,
            self.advances,
            self.declines
        )
        
        self.assertIsInstance(indicator, RegimeIndicator)
        self.assertEqual(indicator.name, "market_breadth")
        self.assertIsInstance(indicator.state, RegimeState)
        self.assertGreaterEqual(indicator.value, 0)
        self.assertLessEqual(indicator.value, 1)
    
    def test_volatility_term_structure(self):
        """Test volatility term structure indicator."""
        indicator = self.detector.vol_indicator.calculate(
            self.implied_vol,
            self.realized_vol
        )
        
        self.assertIsInstance(indicator, RegimeIndicator)
        self.assertEqual(indicator.name, "volatility_term_structure")
        self.assertIsInstance(indicator.state, RegimeState)
    
    def test_correlation_regime(self):
        """Test correlation regime indicator."""
        indicator = self.detector.corr_indicator.calculate(
            self.returns
        )
        
        self.assertIsInstance(indicator, RegimeIndicator)
        self.assertEqual(indicator.name, "correlation_regime")
        self.assertIsInstance(indicator.state, RegimeState)
        self.assertGreaterEqual(indicator.value, 0)
        self.assertLessEqual(indicator.value, 1)
    
    def test_liquidity_regime(self):
        """Test liquidity regime indicator."""
        indicator = self.detector.liq_indicator.calculate(
            self.volume,
            self.spreads,
            self.turnover
        )
        
        self.assertIsInstance(indicator, RegimeIndicator)
        self.assertEqual(indicator.name, "liquidity_regime")
        self.assertIsInstance(indicator.state, RegimeState)
    
    def test_composite_regime(self):
        """Test composite regime detection."""
        market_data = {
            'prices': self.prices,
            'advances': self.advances,
            'declines': self.declines,
            'implied_vol': self.implied_vol,
            'realized_vol': self.realized_vol,
            'returns': self.returns,
            'volume': self.volume,
            'spreads': self.spreads,
            'turnover': self.turnover
        }
        
        regime = self.detector.detect_regime(market_data)
        
        self.assertIn('timestamp', regime)
        self.assertIn('indicators', regime)
        self.assertIn('composite', regime)
        
        # Verify indicator states
        for indicator in regime['indicators'].values():
            self.assertIn('state', indicator)
            self.assertIn('value', indicator)
    
    def test_regime_weights(self):
        """Test regime-specific model weights."""
        # Test default weights
        weights = self.detector.get_regime_weights()
        self.assertIn('xgb', weights)
        self.assertIn('lstm', weights)
        self.assertIn('transformer', weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        
        # Test extreme regime weights
        self.detector.regime_history.append({
            'composite': 'extreme',
            'timestamp': datetime.now()
        })
        weights = self.detector.get_regime_weights()
        self.assertGreater(weights['lstm'], weights['xgb'])
        self.assertGreater(weights['transformer'], weights['xgb'])
    
    def test_regime_history(self):
        """Test regime history tracking."""
        # Generate some regime history
        market_data = {
            'prices': self.prices,
            'advances': self.advances,
            'declines': self.declines,
            'implied_vol': self.implied_vol,
            'realized_vol': self.realized_vol,
            'returns': self.returns,
            'volume': self.volume,
            'spreads': self.spreads,
            'turnover': self.turnover
        }
        
        for _ in range(5):
            self.detector.detect_regime(market_data)
        
        # Get history as DataFrame
        history = self.detector.get_regime_history()
        
        self.assertIsInstance(history, pd.DataFrame)
        self.assertGreater(len(history), 0)
        self.assertIn('timestamp', history.columns)
        self.assertIn('composite', history.columns)
        
        # Verify indicator columns
        for indicator in ['breadth', 'volatility', 'correlation', 'liquidity']:
            self.assertIn(f'{indicator}_state', history.columns)
            self.assertIn(f'{indicator}_value', history.columns)
    
    def test_error_handling(self):
        """Test error handling in regime detection."""
        # Test with missing data
        market_data = {
            'prices': self.prices,
            'advances': self.advances,
            'declines': self.declines
        }
        
        regime = self.detector.detect_regime(market_data)
        self.assertEqual(regime['composite'], 'unknown')
        
        # Test with invalid data
        invalid_prices = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'asset_{i}' for i in range(5)]
        )
        invalid_returns = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'asset_{i}' for i in range(5)]
        )
        
        market_data = {
            'prices': invalid_prices,
            'returns': invalid_returns,
            'advances': self.advances.iloc[:10],
            'declines': self.declines.iloc[:10],
            'implied_vol': self.implied_vol.iloc[:10],
            'realized_vol': self.realized_vol.iloc[:10],
            'volume': self.volume.iloc[:10],
            'spreads': self.spreads.iloc[:10],
            'turnover': self.turnover.iloc[:10]
        }
        
        regime = self.detector.detect_regime(market_data)
        self.assertEqual(regime['composite'], 'unknown')
    
    def test_configuration(self):
        """Test regime detector configuration."""
        config = {
            'ma_short': 20,
            'ma_long': 100,
            'ad_ratio_window': 10,
            'short_term_window': 10,
            'long_term_window': 30,
            'spread_threshold': 0.05,
            'corr_window': 30,
            'spike_threshold': 0.9,
            'dispersion_threshold': 0.2,
            'volume_window': 10,
            'spread_window': 10,
            'turnover_window': 10
        }
        
        detector = RegimeDetector(config)
        
        self.assertEqual(detector.breadth_indicator.ma_short, 20)
        self.assertEqual(detector.breadth_indicator.ma_long, 100)
        self.assertEqual(detector.vol_indicator.spread_threshold, 0.05)
        self.assertEqual(detector.corr_indicator.spike_threshold, 0.9)
        self.assertEqual(detector.liq_indicator.volume_window, 10)

if __name__ == '__main__':
    unittest.main() 
