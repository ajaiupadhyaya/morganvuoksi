"""
Portfolio visualization module with institutional-grade plotting functions.
Provides interactive and static visualizations for portfolio analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def plot_equity_curve(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot portfolio equity curve with optional benchmark overlay.
    
    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns series
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate cumulative returns
    portfolio_curve = (1 + portfolio_returns).cumprod()
    if benchmark_returns is not None:
        benchmark_curve = (1 + benchmark_returns).cumprod()
    
    if interactive:
        fig = go.Figure()
        
        # Add portfolio curve
        fig.add_trace(go.Scatter(
            x=portfolio_curve.index,
            y=portfolio_curve.values,
            name="Portfolio",
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                name="Benchmark",
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio curve
        ax.plot(portfolio_curve.index, portfolio_curve.values,
                label="Portfolio", color='#1f77b4', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            ax.plot(benchmark_curve.index, benchmark_curve.values,
                    label="Benchmark", color='#ff7f0e', linestyle='--', linewidth=2)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax

def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: List[str] = ['sharpe', 'volatility', 'drawdown'],
    title: str = "Rolling Portfolio Metrics",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot rolling portfolio metrics (Sharpe ratio, volatility, drawdown).
    
    Args:
        returns: Portfolio returns series
        window: Rolling window size
        metrics: List of metrics to plot
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate metrics
    rolling_metrics = {}
    
    if 'sharpe' in metrics:
        rolling_metrics['Sharpe Ratio'] = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    
    if 'volatility' in metrics:
        rolling_metrics['Volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    if 'drawdown' in metrics:
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(window, min_periods=1).max()
        rolling_metrics['Drawdown'] = (cum_returns / rolling_max - 1) * 100
    
    if interactive:
        fig = make_subplots(rows=len(metrics), cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05)
        
        for i, (metric_name, metric_data) in enumerate(rolling_metrics.items(), 1):
            fig.add_trace(
                go.Scatter(
                    x=metric_data.index,
                    y=metric_data.values,
                    name=metric_name,
                    line=dict(width=2)
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(metrics),
            template="plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    else:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)),
                                sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, (metric_name, metric_data) in zip(axes, rolling_metrics.items()):
            ax.plot(metric_data.index, metric_data.values, label=metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig, axes

def plot_trade_annotations(
    prices: pd.Series,
    trades: pd.DataFrame,
    title: str = "Price Chart with Trade Annotations",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot price chart with trade entry/exit annotations.
    
    Args:
        prices: Price series
        trades: DataFrame with trade information (timestamp, price, side)
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    if interactive:
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            name="Price",
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add trade markers
        for _, trade in trades.iterrows():
            marker_color = 'green' if trade['side'] == 'buy' else 'red'
            marker_symbol = 'triangle-up' if trade['side'] == 'buy' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[trade['timestamp']],
                y=[trade['price']],
                mode='markers',
                name=f"{trade['side'].title()} {trade['size']}",
                marker=dict(
                    color=marker_color,
                    symbol=marker_symbol,
                    size=10
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price line
        ax.plot(prices.index, prices.values, label="Price", color='#1f77b4')
        
        # Plot trade markers
        for _, trade in trades.iterrows():
            marker_color = 'green' if trade['side'] == 'buy' else 'red'
            marker = '^' if trade['side'] == 'buy' else 'v'
            
            ax.scatter(trade['timestamp'], trade['price'],
                      color=marker_color, marker=marker, s=100,
                      label=f"{trade['side'].title()} {trade['size']}")
        
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax

def plot_risk_heatmap(
    risk_matrix: pd.DataFrame,
    title: str = "Risk Contribution Heatmap",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot risk contribution heatmap.
    
    Args:
        risk_matrix: Risk contribution matrix
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix.values,
            x=risk_matrix.columns,
            y=risk_matrix.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Risk Contribution")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Factor",
            template="plotly_white"
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(risk_matrix, annot=True, cmap='RdYlBu_r',
                    center=0, ax=ax, cbar_kws={'label': 'Risk Contribution'})
        
        ax.set_title(title)
        
        return fig, ax

def plot_strategy_comparison(
    strategy_returns: Dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None,
    metrics: List[str] = ['sharpe', 'volatility', 'max_drawdown'],
    title: str = "Strategy Comparison",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot strategy comparison with performance metrics.
    
    Args:
        strategy_returns: Dictionary of strategy returns
        benchmark_returns: Optional benchmark returns
        metrics: List of metrics to compare
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate metrics for each strategy
    strategy_metrics = {}
    for name, returns in strategy_returns.items():
        metrics_dict = {}
        if 'sharpe' in metrics:
            metrics_dict['Sharpe Ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        if 'volatility' in metrics:
            metrics_dict['Volatility'] = returns.std() * np.sqrt(252)
        if 'max_drawdown' in metrics:
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            metrics_dict['Max Drawdown'] = (cum_returns / rolling_max - 1).min() * 100
        strategy_metrics[name] = metrics_dict
    
    if benchmark_returns is not None:
        benchmark_metrics = {}
        if 'sharpe' in metrics:
            benchmark_metrics['Sharpe Ratio'] = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        if 'volatility' in metrics:
            benchmark_metrics['Volatility'] = benchmark_returns.std() * np.sqrt(252)
        if 'max_drawdown' in metrics:
            cum_returns = (1 + benchmark_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            benchmark_metrics['Max Drawdown'] = (cum_returns / rolling_max - 1).min() * 100
        strategy_metrics['Benchmark'] = benchmark_metrics
    
    if interactive:
        fig = go.Figure()
        
        for strategy, metrics_dict in strategy_metrics.items():
            fig.add_trace(go.Bar(
                name=strategy,
                x=list(metrics_dict.keys()),
                y=list(metrics_dict.values()),
                text=[f"{v:.2f}" for v in metrics_dict.values()],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            barmode='group',
            template="plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(strategy_metrics)
        
        for i, (strategy, metrics_dict) in enumerate(strategy_metrics.items()):
            ax.bar(x + i * width, list(metrics_dict.values()),
                   width, label=strategy)
        
        ax.set_title(title)
        ax.set_xticks(x + width * (len(strategy_metrics) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax 