"""
ML visualization module for monitoring and analyzing ML components.
Provides professional-quality interactive visualizations for institutional use.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_signal_quality_metrics(
    metrics_df: pd.DataFrame,
    title: str = "Signal Quality Metrics Over Time",
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot signal quality metrics over time.
    
    Args:
        metrics_df: DataFrame with columns ['precision', 'auc', 'correlation', 'timestamp']
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    if interactive:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Precision & AUC', 'Forward Return Correlation', 'Signal Decay'),
            vertical_spacing=0.1
        )
        
        # Precision and AUC
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['precision'],
                name='Precision',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['auc'],
                name='AUC',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        # Forward return correlation
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['correlation'],
                name='Correlation',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Signal decay
        decay_cols = [col for col in metrics_df.columns if col.startswith('corr_')]
        for col in decay_cols:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df[col],
                    name=col.replace('corr_', ''),
                    line=dict(dash='dot')
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
        
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Precision and AUC
        axes[0].plot(metrics_df['timestamp'], metrics_df['precision'], label='Precision', color='blue')
        axes[0].plot(metrics_df['timestamp'], metrics_df['auc'], label='AUC', color='green')
        axes[0].set_title('Precision & AUC')
        axes[0].legend()
        
        # Forward return correlation
        axes[1].plot(metrics_df['timestamp'], metrics_df['correlation'], color='red')
        axes[1].set_title('Forward Return Correlation')
        
        # Signal decay
        decay_cols = [col for col in metrics_df.columns if col.startswith('corr_')]
        for col in decay_cols:
            axes[2].plot(metrics_df['timestamp'], metrics_df[col], label=col.replace('corr_', ''), linestyle='--')
        axes[2].set_title('Signal Decay')
        axes[2].legend()
        
        plt.tight_layout()
        return fig

def plot_regime_history(
    regime_history: List[Dict],
    returns: Optional[pd.Series] = None,
    title: str = "Market Regime Analysis",
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot regime history with optional returns overlay.
    
    Args:
        regime_history: List of dicts with 'regime' and 'timestamp' keys
        returns: Optional returns series for overlay
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    regime_df = pd.DataFrame(regime_history)
    regime_df['timestamp'] = pd.to_datetime(regime_df['timestamp'])
    
    if interactive:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Market Regime', 'Cumulative Returns'),
            vertical_spacing=0.1
        )
        
        # Regime plot
        colors = {
            'bullish': 'green',
            'bearish': 'red',
            'neutral': 'gray',
            'high_volatility': 'orange'
        }
        
        for regime in regime_df['regime'].unique():
            mask = regime_df['regime'] == regime
            fig.add_trace(
                go.Scatter(
                    x=regime_df.loc[mask, 'timestamp'],
                    y=[regime] * mask.sum(),
                    mode='markers',
                    name=regime,
                    marker=dict(color=colors.get(regime, 'blue'))
                ),
                row=1, col=1
            )
        
        # Returns plot
        if returns is not None:
            cum_returns = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values,
                    name='Cumulative Returns',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
        
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Regime plot
        colors = {
            'bullish': 'green',
            'bearish': 'red',
            'neutral': 'gray',
            'high_volatility': 'orange'
        }
        
        for regime in regime_df['regime'].unique():
            mask = regime_df['regime'] == regime
            axes[0].scatter(
                regime_df.loc[mask, 'timestamp'],
                [regime] * mask.sum(),
                label=regime,
                color=colors.get(regime, 'blue')
            )
        axes[0].set_title('Market Regime')
        axes[0].legend()
        
        # Returns plot
        if returns is not None:
            cum_returns = (1 + returns).cumprod()
            axes[1].plot(cum_returns.index, cum_returns.values, label='Cumulative Returns')
            axes[1].set_title('Cumulative Returns')
            axes[1].legend()
        
        plt.tight_layout()
        return fig

def plot_feature_importance(
    importance_dict: Dict[str, pd.Series],
    title: str = "Feature Importance Comparison",
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot feature importance comparison across models.
    
    Args:
        importance_dict: Dict mapping model names to feature importance Series
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    # Combine importance scores
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values(importance_df.mean(axis=1), ascending=True)
    
    if interactive:
        fig = go.Figure()
        
        for model in importance_df.columns:
            fig.add_trace(
                go.Bar(
                    y=importance_df.index,
                    x=importance_df[model],
                    name=model,
                    orientation='h'
                )
            )
        
        fig.update_layout(
            title=title,
            barmode='group',
            height=max(400, len(importance_df) * 25),
            template='plotly_white'
        )
        
        return fig
        
    else:
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        importance_df.plot(kind='barh', ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

def plot_model_weights(
    weights_history: List[Dict],
    title: str = "Model Weights by Regime",
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot model weights over time.
    
    Args:
        weights_history: List of dicts with model weights and timestamps
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    weights_df = pd.DataFrame(weights_history)
    weights_df['timestamp'] = pd.to_datetime(weights_df['timestamp'])
    
    if interactive:
        fig = go.Figure()
        
        for model in weights_df.columns:
            if model != 'timestamp':
                fig.add_trace(
                    go.Scatter(
                        x=weights_df['timestamp'],
                        y=weights_df[model],
                        name=model,
                        stackgroup='one'
                    )
                )
        
        fig.update_layout(
            title=title,
            yaxis_title='Weight',
            template='plotly_white'
        )
        
        return fig
        
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        weights_df.set_index('timestamp').plot(
            kind='area',
            stacked=True,
            ax=ax
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig

def plot_signal_decay(
    signals: pd.Series,
    returns: pd.Series,
    windows: List[int] = [1, 5, 10, 20],
    title: str = "Signal Decay Analysis",
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot signal decay analysis across multiple timeframes.
    
    Args:
        signals: Signal series
        returns: Returns series
        windows: List of forward windows to analyze
        title: Plot title
        interactive: Whether to return Plotly (True) or Matplotlib (False) figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    decay_data = []
    for window in windows:
        forward_returns = returns.shift(-window)
        correlation = signals.corr(forward_returns)
        decay_data.append({
            'window': window,
            'correlation': correlation
        })
    
    decay_df = pd.DataFrame(decay_data)
    
    if interactive:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=decay_df['window'],
                y=decay_df['correlation'],
                mode='lines+markers',
                name='Correlation'
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Forward Window (days)',
            yaxis_title='Correlation',
            template='plotly_white'
        )
        
        return fig
        
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(decay_df['window'], decay_df['correlation'], 'o-')
        ax.set_title(title)
        ax.set_xlabel('Forward Window (days)')
        ax.set_ylabel('Correlation')
        plt.tight_layout()
        return fig 