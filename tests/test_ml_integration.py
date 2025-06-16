"""
Integration tests for the ML system with safety features.
Tests the complete pipeline from data ingestion to signal generation.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
import logging
from pathlib import Path

from src.ml.learning_loop import (
    ModelRegistry,
    SignalQualityTracker,
    RegimeDetector,
    LearningLoop
)
from src.ml.safety import (
    ModelSafetyMonitor,
    CrossValidator,
    PositionSizer
)

class TestMLIntegration(unittest.TestCase):
    """Test suite for ML system integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for models
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.test_dir) / 'models'
        self.model_dir.mkdir()
        
        # Initialize components
        self.model_registry = ModelRegistry(str(self.model_dir))
        self.signal_tracker = SignalQualityTracker()
        self.regime_detector = RegimeDetector()
        self.learning_loop = LearningLoop(
            self.model_registry,
            self.signal_tracker,
            self.regime_detector
        )
        
        # Generate test data
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 10
        
        # Generate features
        self.features = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        
        # Generate returns with some predictability
        self.returns = pd.Series(
            np.random.randn(self.n_samples) * 0.1 +
            self.features['feature_0'] * 0.2 +  # Add some signal
            self.features['feature_1'] * 0.1,
            index=self.features.index
        )
        
        # Generate market data
        self.market_data = pd.DataFrame({
            'returns': self.returns,
            'volume': np.random.lognormal(10, 1, self.n_samples),
            'volatility': np.random.gamma(2, 0.1, self.n_samples)
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline with safety features."""
        # Train models
        models = self.learning_loop.retrain_models(
            self.features,
            self.returns,
            self.market_data
        )
        
        # Verify models were trained
        self.assertGreater(len(models), 0)
        
        # Generate signals
        signals, risk_metrics = self.learning_loop.generate_signals(
            self.features,
            self.market_data
        )
        
        # Verify signals
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.features))
        
        # Verify risk metrics
        self.assertIn('position_sizes', risk_metrics)
        self.assertIn('stop_losses', risk_metrics)
        self.assertIn('confidences', risk_metrics)
        
        # Verify signal quality metrics
        metrics = self.signal_tracker.get_metrics()
        self.assertGreater(len(metrics), 0)
        self.assertIn('precision', metrics.columns)
        self.assertIn('auc', metrics.columns)
    
    def test_safety_features(self):
        """Test safety features and risk controls."""
        # Train models
        models = self.learning_loop.retrain_models(
            self.features,
            self.returns,
            self.market_data
        )
        
        # Test circuit breaker
        self.learning_loop.safety_monitor.trigger_circuit_breaker('xgb')
        signals, risk_metrics = self.learning_loop.generate_signals(
            self.features,
            self.market_data
        )
        self.assertNotIn('xgb', risk_metrics['confidences'])
        
        # Test performance degradation
        self.learning_loop.safety_monitor.performance_history['xgb'] = [
            {'timestamp': datetime.now(), 'score': 0.3}  # Below threshold
        ]
        signals, risk_metrics = self.learning_loop.generate_signals(
            self.features,
            self.market_data
        )
        self.assertNotIn('xgb', risk_metrics['confidences'])
        
        # Test position sizing
        for model_id, size in risk_metrics['position_sizes'].items():
            self.assertGreaterEqual(size, 0)
            self.assertLessEqual(size, 1)
        
        # Test stop-losses
        for model_id, stop_loss in risk_metrics['stop_losses'].items():
            self.assertLess(stop_loss, 0)  # Stop-loss should be negative
    
    def test_regime_switching(self):
        """Test regime detection and model weight adjustment."""
        # Generate different return patterns
        high_vol_returns = self.returns * 2  # High volatility
        trend_returns = pd.Series(
            np.cumsum(np.random.randn(self.n_samples) * 0.1),
            index=self.returns.index
        )  # Trending
        
        # Test high volatility regime
        self.regime_detector.detect_regime(high_vol_returns)
        weights = self.regime_detector.get_regime_weights()
        self.assertIn('high_volatility', self.regime_detector.regime_history[-1]['regime'])
        
        # Test trending regime
        self.regime_detector.detect_regime(trend_returns)
        weights = self.regime_detector.get_regime_weights()
        self.assertIn(
            self.regime_detector.regime_history[-1]['regime'],
            ['bullish', 'bearish']
        )
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test with invalid data
        invalid_features = pd.DataFrame(
            np.random.randn(10, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        invalid_returns = pd.Series(np.random.randn(10))
        
        # Should handle insufficient data gracefully
        models = self.learning_loop.retrain_models(
            invalid_features,
            invalid_returns,
            self.market_data
        )
        self.assertEqual(len(models), 0)
        
        # Test with missing market data
        signals, risk_metrics = self.learning_loop.generate_signals(
            self.features,
            None
        )
        self.assertIsInstance(signals, pd.Series)
        self.assertIsInstance(risk_metrics, dict)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train and save models
        models = self.learning_loop.retrain_models(
            self.features,
            self.returns,
            self.market_data
        )
        
        # Verify models were saved
        for model_id in models.keys():
            model_path = self.model_dir / f"{model_id}.joblib"
            self.assertTrue(model_path.exists())
        
        # Load models and verify
        for model_id in models.keys():
            model, metadata = self.model_registry.load_model(model_id)
            self.assertIsNotNone(model)
            self.assertIn('regime', metadata)
            self.assertIn('n_samples', metadata)
            self.assertIn('feature_importance', metadata)
            self.assertIn('cross_val_score', metadata)

if __name__ == '__main__':
    unittest.main() 
