"""
Unit tests for the visualization module.
"""

import pytest
import pandas as pd
import numpy as np
from src.visuals import (
    plot_equity_curve,
    plot_rolling_metrics,
    plot_trade_annotations,
    plot_risk_heatmap,
    plot_strategy_comparison,
    plot_signal_strength,
    plot_feature_importance,
    plot_signal_decay,
    plot_risk_decomposition,
    plot_correlation_matrix,
    plot_drawdown_analysis
)

@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    dates = pd.date_range('2023-01-01', periods=100)
    returns = pd.Series(np.random.normal(0.0001, 0.02, 100), index=dates)
    return returns

@pytest.fixture
def sample_signals():
    """Generate sample signals for testing."""
    dates = pd.date_range('2023-01-01', periods=100)
    signals = pd.DataFrame({
        'AAPL': np.random.randn(100),
        'MSFT': np.random.randn(100),
        'GOOG': np.random.randn(100)
    }, index=dates)
    return signals

@pytest.fixture
def sample_prices():
    """Generate sample prices for testing."""
    dates = pd.date_range('2023-01-01', periods=100)
    prices = pd.DataFrame({
        'AAPL': np.random.lognormal(5, 0.1, 100),
        'MSFT': np.random.lognormal(5, 0.1, 100),
        'GOOG': np.random.lognormal(5, 0.1, 100)
    }, index=dates)
    return prices

@pytest.fixture
def sample_trades():
    """Generate sample trades for testing."""
    dates = pd.date_range('2023-01-01', periods=10)
    trades = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.lognormal(5, 0.1, 10),
        'side': np.random.choice(['buy', 'sell'], 10),
        'size': np.random.randint(1, 100, 10)
    })
    return trades

def test_plot_equity_curve(sample_returns):
    """Test equity curve plotting."""
    fig = plot_equity_curve(sample_returns)
    assert fig is not None

def test_plot_rolling_metrics(sample_returns):
    """Test rolling metrics plotting."""
    fig = plot_rolling_metrics(sample_returns)
    assert fig is not None

def test_plot_trade_annotations(sample_prices, sample_trades):
    """Test trade annotations plotting."""
    fig = plot_trade_annotations(sample_prices['AAPL'], sample_trades)
    assert fig is not None

def test_plot_risk_heatmap(sample_signals):
    """Test risk heatmap plotting."""
    risk_matrix = sample_signals.corr()
    fig = plot_risk_heatmap(risk_matrix)
    assert fig is not None

def test_plot_strategy_comparison(sample_returns):
    """Test strategy comparison plotting."""
    strategies = {
        'Strategy 1': sample_returns,
        'Strategy 2': sample_returns * 1.1
    }
    fig = plot_strategy_comparison(strategies)
    assert fig is not None

def test_plot_signal_strength(sample_signals, sample_prices):
    """Test signal strength plotting."""
    fig = plot_signal_strength(sample_signals, sample_prices)
    assert fig is not None

def test_plot_feature_importance(sample_signals):
    """Test feature importance plotting."""
    importance = pd.Series(
        np.random.rand(len(sample_signals.columns)),
        index=sample_signals.columns
    )
    fig = plot_feature_importance(importance)
    assert fig is not None

def test_plot_signal_decay(sample_signals, sample_returns):
    """Test signal decay plotting."""
    fig = plot_signal_decay(sample_signals, sample_returns)
    assert fig is not None

def test_plot_risk_decomposition(sample_signals):
    """Test risk decomposition plotting."""
    risk_contrib = pd.DataFrame(
        np.random.rand(3, 3),
        index=['Factor 1', 'Factor 2', 'Factor 3'],
        columns=['AAPL', 'MSFT', 'GOOG']
    )
    fig = plot_risk_decomposition(risk_contrib)
    assert fig is not None

def test_plot_correlation_matrix(sample_signals):
    """Test correlation matrix plotting."""
    fig = plot_correlation_matrix(sample_signals)
    assert fig is not None

def test_plot_drawdown_analysis(sample_returns):
    """Test drawdown analysis plotting."""
    fig = plot_drawdown_analysis(sample_returns)
    assert fig is not None 