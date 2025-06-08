import pytest
import numpy as np

def test_strategy_consistency(sample_signals):
    # Example: signals should not be all zeros
    assert not np.allclose(sample_signals.values, 0)

def test_risk_guardrails(sample_signals):
    # Example: volatility should not exceed threshold
    vol = sample_signals.std().mean()
    assert vol < 5  # Example threshold

def test_performance_threshold():
    # Placeholder: replace with real performance metric
    sharpe = 1.5  # Example value
    assert sharpe > 1.0 