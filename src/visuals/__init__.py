"""
Institutional-grade visualization module for quantitative analysis.
Provides interactive and static visualizations for portfolio analysis,
signal strength, risk metrics, and strategy comparison.
"""

from .portfolio_visuals import (
    plot_equity_curve,
    plot_rolling_metrics,
    plot_trade_annotations,
    plot_risk_heatmap,
    plot_strategy_comparison
)

from .signal_visuals import (
    plot_signal_strength,
    plot_feature_importance,
    plot_signal_decay
)

from .risk_visuals import (
    plot_risk_decomposition,
    plot_correlation_matrix,
    plot_drawdown_analysis
)

__all__ = [
    'plot_equity_curve',
    'plot_rolling_metrics',
    'plot_trade_annotations',
    'plot_risk_heatmap',
    'plot_strategy_comparison',
    'plot_signal_strength',
    'plot_feature_importance',
    'plot_signal_decay',
    'plot_risk_decomposition',
    'plot_correlation_matrix',
    'plot_drawdown_analysis'
] 