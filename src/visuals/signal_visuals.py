"""
Signal visualization module with institutional-grade plotting functions.
Provides interactive and static visualizations for signal analysis.
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

def plot_signal_strength(
    signals: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    title: str = "Signal Strength Analysis",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot signal strength analysis with optional price overlay.
    
    Args:
        signals: Signal DataFrame
        prices: Optional price DataFrame
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    if interactive:
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05)
        
        # Plot signals
        for col in signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals[col],
                    name=f"Signal: {col}",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot prices if provided
        if prices is not None:
            for col in prices.columns:
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices[col],
                        name=f"Price: {col}",
                        line=dict(width=2, dash='dash')
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=title,
            height=800,
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
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot signals
        for col in signals.columns:
            axes[0].plot(signals.index, signals[col],
                        label=f"Signal: {col}", linewidth=2)
        
        axes[0].set_title("Signal Strength")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot prices if provided
        if prices is not None:
            for col in prices.columns:
                axes[1].plot(prices.index, prices[col],
                            label=f"Price: {col}", linestyle='--', linewidth=2)
        
        axes[1].set_title("Price")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig, axes

def plot_feature_importance(
    feature_importance: pd.Series,
    title: str = "Feature Importance",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot feature importance analysis.
    
    Args:
        feature_importance: Feature importance series
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Sort features by importance
    feature_importance = feature_importance.sort_values(ascending=True)
    
    if interactive:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation='h',
            text=[f"{v:.2f}" for v in feature_importance.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white"
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feature_importance.plot(kind='barh', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.grid(True, alpha=0.3)
        
        return fig, ax

def plot_signal_decay(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    windows: List[int] = [1, 5, 10, 20],
    title: str = "Signal Decay Analysis",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot signal decay analysis.
    
    Args:
        signals: Signal DataFrame
        returns: Returns DataFrame
        windows: List of forward-looking windows
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate forward returns for each window
    forward_returns = {}
    for window in windows:
        forward_returns[window] = returns.shift(-window)
    
    # Calculate signal-return correlations
    correlations = {}
    for window in windows:
        correlations[window] = signals.corrwith(forward_returns[window])
    
    if interactive:
        fig = go.Figure()
        
        for window in windows:
            fig.add_trace(go.Scatter(
                x=correlations[window].index,
                y=correlations[window].values,
                name=f"{window}-day",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Signal",
            yaxis_title="Correlation",
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
        
        for window in windows:
            ax.plot(correlations[window].index,
                   correlations[window].values,
                   label=f"{window}-day", linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel("Signal")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax 
