"""
Tests for the ML learning loop module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from src.ml.learning_loop import (
    ModelRegistry,
    SignalQualityTracker,
    RegimeDetector,
    LearningLoop,
    LSTMClassifier,
    TransformerClassifier
)

class TestModelRegistry(unittest.TestCase):
    """Test the ModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(model_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_model(self):
        """Test saving and loading models."""
        # Create test model
        model = LSTMClassifier(input_dim=10, hidden_dim=64, output_dim=1)
        metadata = {'test': 'metadata'}
        
        # Save model
        self.registry.save_model('test_model', model, metadata)
        
        # Load model
        loaded_model, loaded_metadata = self.registry.load_model('test_model')
        
        # Verify
        self.assertIsInstance(loaded_model, LSTMClassifier)
        self.assertEqual(loaded_metadata['test'], 'metadata')
    
    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        with self.assertRaises(ValueError):
            self.registry.save_model('invalid', object(), {})

class TestSignalQualityTracker(unittest.TestCase):
    """Test the SignalQualityTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.tracker = SignalQualityTracker(window=5)
    
    def test_update_metrics(self):
        """Test updating signal quality metrics."""
        # Create test data
        signals = pd.Series([0.8, 0.2, 0.9, 0.1, 0.7])
        returns = pd.Series([1, -1, 1, -1, 1])
        
        # Update metrics
        metrics = self.tracker.update(signals, returns)
        
        # Verify
        self.assertIn('precision', metrics)
        self.assertIn('auc', metrics)
        self.assertIn('correlation', metrics)
    
    def test_get_metrics(self):
        """Test getting historical metrics."""
        # Add some metrics
        for _ in range(3):
            self.tracker.update(
                pd.Series([0.8, 0.2, 0.9]),
                pd.Series([1, -1, 1])
            )
        
        # Get metrics
        metrics_df = self.tracker.get_metrics()
        
        # Verify
        self.assertEqual(len(metrics_df), 3)
        self.assertIn('precision', metrics_df.columns)

class TestRegimeDetector(unittest.TestCase):
    """Test the RegimeDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = RegimeDetector(window=5)
    
    def test_detect_regime(self):
        """Test regime detection."""
        # Create test data
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        
        # Detect regime
        regime = self.detector.detect_regime(returns)
        
        # Verify
        self.assertIn(regime, ['bullish', 'bearish', 'neutral', 'high_volatility'])
    
    def test_get_regime_weights(self):
        """Test getting regime-specific weights."""
        # Add some regime history
        self.detector.regime_history = [
            {'regime': 'bullish', 'timestamp': datetime.now()}
        ]
        
        # Get weights
        weights = self.detector.get_regime_weights()
        
        # Verify
        self.assertIn('xgb', weights)
        self.assertIn('lstm', weights)
        self.assertIn('transformer', weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

class TestLearningLoop(unittest.TestCase):
    """Test the LearningLoop class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(model_dir=self.temp_dir)
        self.tracker = SignalQualityTracker()
        self.detector = RegimeDetector()
        self.loop = LearningLoop(
            self.registry,
            self.tracker,
            self.detector,
            retrain_interval=1,
            min_samples=10
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_should_retrain(self):
        """Test retraining decision logic."""
        # Should retrain initially
        self.assertTrue(self.loop.should_retrain())
        
        # Set last retrain
        self.loop.last_retrain = datetime.now()
        self.assertFalse(self.loop.should_retrain())
        
        # Set last retrain to past
        self.loop.last_retrain = datetime.now() - timedelta(days=2)
        self.assertTrue(self.loop.should_retrain())
    
    def test_retrain_models(self):
        """Test model retraining."""
        # Create test data
        features = pd.DataFrame(np.random.randn(20, 10))
        returns = pd.Series(np.random.randn(20))
        
        # Retrain models
        models = self.loop.retrain_models(features, returns)
        
        # Verify
        self.assertIn('xgb', models)
        self.assertIn('lstm', models)
        self.assertIn('transformer', models)
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Create test data
        features = pd.DataFrame(np.random.randn(10, 10))
        market_data = pd.DataFrame({'returns': np.random.randn(10)})
        
        # Train models first
        self.loop.retrain_models(features, market_data['returns'])
        
        # Generate signals
        signals = self.loop.generate_signals(features, market_data)
        
        # Verify
        self.assertEqual(len(signals), len(features))
        self.assertTrue(all(0 <= x <= 1 for x in signals))

if __name__ == '__main__':
    unittest.main() 
