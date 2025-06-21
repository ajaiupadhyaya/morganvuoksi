"""
Institutional-grade visualization module for quantitative analysis.
Provides interactive and static visualizations for portfolio analysis,
signal strength, risk metrics, and strategy comparison.
"""

from .charting import (
    create_candlestick_chart,
    create_technical_chart,
    create_prediction_chart,
    create_loss_curve,
    create_feature_importance_chart,
    create_sentiment_chart,
    create_portfolio_chart,
    create_efficient_frontier_chart,
    create_risk_dashboard
)

__all__ = [
    "create_candlestick_chart",
    "create_technical_chart",
    "create_prediction_chart",
    "create_loss_curve",
    "create_feature_importance_chart",
    "create_sentiment_chart",
    "create_portfolio_chart",
    "create_efficient_frontier_chart",
    "create_risk_dashboard"
] 
