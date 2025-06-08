import pytest
from src.data import market_data
from src.signals import signal_generator
from src.portfolio import optimizer
from src.execution import simulate

def test_end_to_end_pipeline(sample_market_data, sample_signals):
    # Simulate data ingestion
    data = sample_market_data
    # Simulate signal generation
    signals = sample_signals
    # Simulate portfolio optimization
    # Placeholder: replace with real optimizer call
    portfolio = signals.mean(axis=1)
    # Simulate execution
    # Placeholder: replace with real execution call
    executed = portfolio.apply(lambda x: x * 0.99)
    assert executed is not None
    assert not executed.isnull().any() 