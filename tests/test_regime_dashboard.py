"""
Tests for regime dashboard visualization module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import logging

from src.visuals.regime_dashboard import RegimeDashboard

class TestRegimeDashboard(unittest.TestCase):
    """Test suite for regime dashboard visualization."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test data
        self.start_date = datetime(2023, 1, 1)
        self.dates = pd.date_range(self.start_date, periods=100, freq='D')
        
        # Create regime history
        self.regime_history = pd.DataFrame({
            'timestamp': self.dates,
            'breadth_value': np.random.normal(0, 1, 100),
            'breadth_state': np.random.choice(['high', 'neutral', 'low'], 100),
            'volatility_value': np.random.normal(0, 1, 100),
            'volatility_state': np.random.choice(['high', 'neutral', 'low'], 100),
            'correlation_value': np.random.normal(0, 1, 100),
            'correlation_state': np.random.choice(['high', 'neutral', 'low'], 100),
            'liquidity_value': np.random.normal(0, 1, 100),
            'liquidity_state': np.random.choice(['high', 'neutral', 'low'], 100),
            'composite': np.random.choice(['extreme', 'high', 'neutral', 'low'], 100)
        })
        
        # Create portfolio equity
        self.portfolio_equity = pd.Series(
            np.cumsum(np.random.normal(0, 1, 100)),
            index=self.dates
        )
        
        # Create model performance data
        self.model_performance = {
            'extreme': {'sharpe': 1.5, 'precision': 0.7, 'returns': 0.1},
            'high': {'sharpe': 1.2, 'precision': 0.65, 'returns': 0.08},
            'neutral': {'sharpe': 0.8, 'precision': 0.6, 'returns': 0.05},
            'low': {'sharpe': 0.5, 'precision': 0.55, 'returns': 0.03}
        }
        
        # Create signal quality data
        self.signal_quality = {
            'extreme': {
                'decay': pd.Series(np.exp(-np.linspace(0, 5, 20))),
                'precision': 0.7,
                'recall': 0.6,
                'f1': 0.65
            },
            'high': {
                'decay': pd.Series(np.exp(-np.linspace(0, 5, 20))),
                'precision': 0.65,
                'recall': 0.55,
                'f1': 0.6
            },
            'neutral': {
                'decay': pd.Series(np.exp(-np.linspace(0, 5, 20))),
                'precision': 0.6,
                'recall': 0.5,
                'f1': 0.55
            },
            'low': {
                'decay': pd.Series(np.exp(-np.linspace(0, 5, 20))),
                'precision': 0.55,
                'recall': 0.45,
                'f1': 0.5
            }
        }
        
        # Create temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize dashboard
        self.dashboard = RegimeDashboard(
            regime_history=self.regime_history,
            portfolio_equity=self.portfolio_equity,
            model_performance=self.model_performance,
            signal_quality=self.signal_quality
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_core_indicators_plot(self):
        """Test core indicators plot generation."""
        # Test interactive plot
        fig = self.dashboard.plot_core_indicators(interactive=True)
        self.assertIsNotNone(fig)
        
        # Test static plot
        fig = self.dashboard.plot_core_indicators(interactive=False)
        self.assertIsNotNone(fig)
        
        # Test plot saving
        self.dashboard.plot_core_indicators(
            interactive=True,
            save_path=str(Path(self.temp_dir) / 'core_indicators.html')
        )
        self.assertTrue(Path(self.temp_dir, 'core_indicators.html').exists())
    
    def test_composite_regime_plot(self):
        """Test composite regime plot generation."""
        # Test interactive plot
        fig = self.dashboard.plot_composite_regime(interactive=True)
        self.assertIsNotNone(fig)
        
        # Test static plot
        fig = self.dashboard.plot_composite_regime(interactive=False)
        self.assertIsNotNone(fig)
        
        # Test plot saving
        self.dashboard.plot_composite_regime(
            interactive=True,
            save_path=str(Path(self.temp_dir) / 'composite_regime.html')
        )
        self.assertTrue(Path(self.temp_dir, 'composite_regime.html').exists())
    
    def test_model_performance_plot(self):
        """Test model performance plot generation."""
        # Test interactive plot
        fig = self.dashboard.plot_model_performance(interactive=True)
        self.assertIsNotNone(fig)
        
        # Test static plot
        fig = self.dashboard.plot_model_performance(interactive=False)
        self.assertIsNotNone(fig)
        
        # Test plot saving
        self.dashboard.plot_model_performance(
            interactive=True,
            save_path=str(Path(self.temp_dir) / 'model_performance.html')
        )
        self.assertTrue(Path(self.temp_dir, 'model_performance.html').exists())
    
    def test_signal_quality_plot(self):
        """Test signal quality plot generation."""
        # Test interactive plot
        fig = self.dashboard.plot_signal_quality(interactive=True)
        self.assertIsNotNone(fig)
        
        # Test static plot
        fig = self.dashboard.plot_signal_quality(interactive=False)
        self.assertIsNotNone(fig)
        
        # Test plot saving
        self.dashboard.plot_signal_quality(
            interactive=True,
            save_path=str(Path(self.temp_dir) / 'signal_quality.html')
        )
        self.assertTrue(Path(self.temp_dir, 'signal_quality.html').exists())
    
    def test_full_dashboard_creation(self):
        """Test creation of full dashboard."""
        # Create dashboard
        self.dashboard.create_dashboard(
            output_dir=self.temp_dir,
            interactive=True
        )
        
        # Check all files were created
        expected_files = [
            'core_indicators.html',
            'composite_regime.html',
            'model_performance.html',
            'signal_quality.html'
        ]
        
        for file in expected_files:
            self.assertTrue(Path(self.temp_dir, file).exists())
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create dashboard without optional data
        dashboard = RegimeDashboard(regime_history=self.regime_history)
        
        # Test plots still work
        fig = dashboard.plot_core_indicators(interactive=True)
        self.assertIsNotNone(fig)
        
        fig = dashboard.plot_composite_regime(interactive=True)
        self.assertIsNotNone(fig)
        
        # Test plots with missing data return None
        fig = dashboard.plot_model_performance(interactive=True)
        self.assertIsNone(fig)
        
        fig = dashboard.plot_signal_quality(interactive=True)
        self.assertIsNone(fig)

if __name__ == '__main__':
    unittest.main() 
