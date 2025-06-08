"""
Risk visualization module with institutional-grade plotting functions.
Provides interactive and static visualizations for risk analysis.
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

def plot_risk_decomposition(
    risk_contributions: pd.DataFrame,
    title: str = "Risk Decomposition",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot risk decomposition analysis.
    
    Args:
        risk_contributions: Risk contribution DataFrame
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    if interactive:
        fig = go.Figure()
        
        for col in risk_contributions.columns:
            fig.add_trace(go.Bar(
                x=risk_contributions.index,
                y=risk_contributions[col],
                name=col,
                text=[f"{v:.1%}" for v in risk_contributions[col]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Risk Contribution",
            barmode='stack',
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
        
        risk_contributions.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Risk Contribution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax

def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Correlation Matrix",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot correlation matrix heatmap.
    
    Args:
        returns: Returns DataFrame
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu',
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title)
        
        return fig, ax

def plot_drawdown_analysis(
    returns: pd.Series,
    window: int = 252,
    title: str = "Drawdown Analysis",
    interactive: bool = True
) -> Union[go.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot drawdown analysis.
    
    Args:
        returns: Returns series
        window: Rolling window size
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly figure or Matplotlib figure and axes
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate drawdown
    rolling_max = cum_returns.rolling(window, min_periods=1).max()
    drawdown = (cum_returns / rolling_max - 1) * 100
    
    if interactive:
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05)
        
        # Plot cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                name="Cumulative Returns",
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Plot drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line=dict(color='#ff7f0e', width=2)
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
        
        # Plot cumulative returns
        axes[0].plot(cum_returns.index, cum_returns.values,
                    label="Cumulative Returns", color='#1f77b4', linewidth=2)
        axes[0].set_title("Cumulative Returns")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot drawdown
        axes[1].plot(drawdown.index, drawdown.values,
                    label="Drawdown", color='#ff7f0e', linewidth=2)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig, axes 