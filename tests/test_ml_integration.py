"""
Integration tests for the ML system.
Verifies ML components work together and integrate with the trading system.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import logging

from src.ml.learning_loop import (
    ModelRegistry,
    SignalQualityTracker,
    RegimeDetector,
    LearningLoop
)
from src.visuals.ml_visuals import (
    plot_signal_quality_metrics,
    plot_regime_history,
    plot_feature_importance,
    plot_model_weights,
    plot_signal_decay
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestMLIntegration(unittest.TestCase):
    """Test ML system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(model_dir=self.temp_dir)
        self.tracker = SignalQualityTracker(window=5)
        self.detector = RegimeDetector(window=5)
        self.loop = LearningLoop(
            self.registry,
            self.tracker,
            self.detector,
            retrain_interval=1,
            min_samples=10
        )
        
        # Generate test data
        np.random.seed(42)
        self.features = pd.DataFrame(
            np.random.randn(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        self.returns = pd.Series(np.random.randn(100))
        self.market_data = pd.DataFrame({
            'returns': self.returns,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_ml_pipeline(self):
        """Test the complete ML pipeline from training to signal generation."""
        # Initial training
        models = self.loop.retrain_models(
            self.features,
            self.returns,
            self.market_data
        )
        self.assertIn('xgb', models)
        self.assertIn('lstm', models)
        self.assertIn('transformer', models)
        
        # Generate signals
        signals = self.loop.generate_signals(
            self.features,
            self.market_data
        )
        self.assertEqual(len(signals), len(self.features))
        self.assertTrue(all(0 <= x <= 1 for x in signals))
        
        # Check signal quality metrics
        metrics = self.tracker.get_metrics()
        self.assertGreater(len(metrics), 0)
        self.assertIn('precision', metrics.columns)
        self.assertIn('auc', metrics.columns)
        
        # Check regime detection
        regime = self.detector.detect_regime(self.returns)
        self.assertIn(regime, ['bullish', 'bearish', 'neutral', 'high_volatility'])
        
        # Check model weights
        weights = self.detector.get_regime_weights()
        self.assertIn('xgb', weights)
        self.assertIn('lstm', weights)
        self.assertIn('transformer', weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
    
    def test_retraining_schedule(self):
        """Test model retraining schedule."""
        # Initial training
        self.loop.retrain_models(self.features, self.returns)
        self.assertIsNotNone(self.loop.last_retrain)
        
        # Should not retrain immediately
        self.assertFalse(self.loop.should_retrain())
        
        # Force retrain by setting last_retrain to past
        self.loop.last_retrain = datetime.now() - timedelta(days=2)
        self.assertTrue(self.loop.should_retrain())
    
    def test_regime_switching(self):
        """Test regime switching and model weight adjustment."""
        # Train models
        self.loop.retrain_models(self.features, self.returns)
        
        # Generate different return patterns
        bullish_returns = pd.Series(np.random.uniform(0.001, 0.01, 100))
        bearish_returns = pd.Series(np.random.uniform(-0.01, -0.001, 100))
        volatile_returns = pd.Series(np.random.uniform(-0.02, 0.02, 100))
        
        # Check regime detection
        bullish_regime = self.detector.detect_regime(bullish_returns)
        bearish_regime = self.detector.detect_regime(bearish_returns)
        volatile_regime = self.detector.detect_regime(volatile_returns)
        
        self.assertEqual(bullish_regime, 'bullish')
        self.assertEqual(bearish_regime, 'bearish')
        self.assertEqual(volatile_regime, 'high_volatility')
        
        # Check weight adjustment
        bullish_weights = self.detector.get_regime_weights()
        self.assertGreater(bullish_weights['xgb'], 0.3)  # XGBoost favored in bullish
        
        bearish_weights = self.detector.get_regime_weights()
        self.assertGreater(bearish_weights['transformer'], 0.3)  # Transformer favored in bearish
    
    def test_signal_quality_tracking(self):
        """Test signal quality tracking and visualization."""
        # Generate signals
        signals = self.loop.generate_signals(self.features, self.market_data)
        
        # Update metrics
        metrics = self.tracker.update(signals, self.returns)
        self.assertIn('precision', metrics)
        self.assertIn('auc', metrics)
        self.assertIn('correlation', metrics)
        
        # Test visualization
        metrics_df = self.tracker.get_metrics()
        fig = plot_signal_quality_metrics(metrics_df)
        self.assertIsNotNone(fig)
    
    def test_feature_importance_tracking(self):
        """Test feature importance tracking and visualization."""
        # Train models
        models = self.loop.retrain_models(self.features, self.returns)
        
        # Get feature importance
        importance_dict = {}
        for model_id, model in models.items():
            importance_dict[model_id] = self.loop._get_feature_importance(
                model,
                self.features
            )
        
        # Test visualization
        fig = plot_feature_importance(importance_dict)
        self.assertIsNotNone(fig)
    
    def test_error_handling(self):
        """Test error handling in ML pipeline."""
        # Test with missing data
        with self.assertRaises(Exception):
            self.loop.generate_signals(pd.DataFrame(), None)
        
        # Test with invalid model
        with self.assertRaises(ValueError):
            self.registry.save_model('invalid', object(), {})
        
        # Test with insufficient samples
        small_features = pd.DataFrame(np.random.randn(5, 10))
        small_returns = pd.Series(np.random.randn(5))
        models = self.loop.retrain_models(small_features, small_returns)
        self.assertEqual(len(models), 0)  # Should return empty dict
    
    def test_visualization_integration(self):
        """Test integration of visualization components."""
        # Generate test data
        signals = self.loop.generate_signals(self.features, self.market_data)
        regime_history = self.detector.regime_history
        weights_history = [
            {
                'timestamp': datetime.now() - timedelta(days=i),
                'xgb': 0.4,
                'lstm': 0.3,
                'transformer': 0.3
            }
            for i in range(10)
        ]
        
        # Test all visualizations
        metrics_fig = plot_signal_quality_metrics(self.tracker.get_metrics())
        regime_fig = plot_regime_history(regime_history, self.returns)
        weights_fig = plot_model_weights(weights_history)
        decay_fig = plot_signal_decay(signals, self.returns)
        
        self.assertIsNotNone(metrics_fig)
        self.assertIsNotNone(regime_fig)
        self.assertIsNotNone(weights_fig)
        self.assertIsNotNone(decay_fig)

if __name__ == '__main__':
    unittest.main() 