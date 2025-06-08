"""
Visualization functions for regime detection system.
Provides interactive and static plots for market regimes and indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def plot_regime_indicators(
    regime_history: pd.DataFrame,
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot regime indicators over time.
    
    Args:
        regime_history: DataFrame with regime history
        interactive: Whether to return Plotly or Matplotlib figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    if interactive:
        # Create subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                'Market Breadth',
                'Volatility Term Structure',
                'Correlation Regime',
                'Liquidity Regime'
            ),
            vertical_spacing=0.1
        )
        
        # Add traces for each indicator
        indicators = ['breadth', 'volatility', 'correlation', 'liquidity']
        colors = ['blue', 'red', 'green', 'purple']
        
        for i, (indicator, color) in enumerate(zip(indicators, colors), 1):
            # Add value line
            fig.add_trace(
                go.Scatter(
                    x=regime_history['timestamp'],
                    y=regime_history[f'{indicator}_value'],
                    name=f'{indicator.title()} Value',
                    line=dict(color=color)
                ),
                row=i,
                col=1
            )
            
            # Add state markers
            states = regime_history[f'{indicator}_state'].unique()
            for state in states:
                mask = regime_history[f'{indicator}_state'] == state
                fig.add_trace(
                    go.Scatter(
                        x=regime_history.loc[mask, 'timestamp'],
                        y=regime_history.loc[mask, f'{indicator}_value'],
                        mode='markers',
                        name=f'{indicator.title()} - {state}',
                        marker=dict(
                            color=color,
                            symbol='circle',
                            size=8
                        )
                    ),
                    row=i,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text='Regime Indicators Over Time'
        )
        
        return fig
        
    else:
        # Create Matplotlib figure
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        fig.suptitle('Regime Indicators Over Time')
        
        # Plot each indicator
        indicators = ['breadth', 'volatility', 'correlation', 'liquidity']
        colors = ['blue', 'red', 'green', 'purple']
        
        for i, (indicator, color) in enumerate(zip(indicators, colors)):
            ax = axes[i]
            
            # Plot value line
            ax.plot(
                regime_history['timestamp'],
                regime_history[f'{indicator}_value'],
                color=color,
                label=f'{indicator.title()} Value'
            )
            
            # Plot state markers
            states = regime_history[f'{indicator}_state'].unique()
            for state in states:
                mask = regime_history[f'{indicator}_state'] == state
                ax.scatter(
                    regime_history.loc[mask, 'timestamp'],
                    regime_history.loc[mask, f'{indicator}_value'],
                    color=color,
                    label=f'{indicator.title()} - {state}'
                )
            
            ax.set_title(f'{indicator.title()} Indicator')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        return fig

def plot_composite_regime(
    regime_history: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot composite regime with optional returns overlay.
    
    Args:
        regime_history: DataFrame with regime history
        returns: Optional returns series to overlay
        interactive: Whether to return Plotly or Matplotlib figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    if interactive:
        # Create figure
        fig = go.Figure()
        
        # Add regime state markers
        regimes = regime_history['composite'].unique()
        colors = {
            'extreme': 'red',
            'high': 'orange',
            'neutral': 'green',
            'low': 'blue',
            'unknown': 'gray'
        }
        
        for regime in regimes:
            mask = regime_history['composite'] == regime
            fig.add_trace(
                go.Scatter(
                    x=regime_history.loc[mask, 'timestamp'],
                    y=[regime] * mask.sum(),
                    mode='markers',
                    name=regime,
                    marker=dict(
                        color=colors.get(regime, 'gray'),
                        symbol='circle',
                        size=10
                    )
                )
            )
        
        # Add returns overlay if provided
        if returns is not None:
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns,
                    name='Returns',
                    line=dict(color='black', width=1),
                    yaxis='y2'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Composite Regime with Returns Overlay',
            yaxis=dict(
                title='Regime',
                categoryorder='array',
                categoryarray=['extreme', 'high', 'neutral', 'low', 'unknown']
            ),
            yaxis2=dict(
                title='Returns',
                overlaying='y',
                side='right'
            ),
            showlegend=True
        )
        
        return fig
        
    else:
        # Create Matplotlib figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot regime states
        regimes = regime_history['composite'].unique()
        colors = {
            'extreme': 'red',
            'high': 'orange',
            'neutral': 'green',
            'low': 'blue',
            'unknown': 'gray'
        }
        
        for regime in regimes:
            mask = regime_history['composite'] == regime
            ax1.scatter(
                regime_history.loc[mask, 'timestamp'],
                [regime] * mask.sum(),
                color=colors.get(regime, 'gray'),
                label=regime
            )
        
        # Add returns overlay if provided
        if returns is not None:
            ax2 = ax1.twinx()
            ax2.plot(returns.index, returns, 'k-', label='Returns', alpha=0.5)
            ax2.set_ylabel('Returns')
        
        # Update layout
        ax1.set_title('Composite Regime with Returns Overlay')
        ax1.set_ylabel('Regime')
        ax1.legend()
        ax1.grid(True)
        
        plt.tight_layout()
        return fig

def plot_regime_transitions(
    regime_history: pd.DataFrame,
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot regime transition matrix and heatmap.
    
    Args:
        regime_history: DataFrame with regime history
        interactive: Whether to return Plotly or Matplotlib figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    # Calculate transition matrix
    transitions = pd.crosstab(
        regime_history['composite'].shift(),
        regime_history['composite']
    )
    
    # Normalize rows
    transitions = transitions.div(transitions.sum(axis=1), axis=0)
    
    if interactive:
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transitions.values,
            x=transitions.columns,
            y=transitions.index,
            colorscale='RdYlGn',
            text=transitions.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        # Update layout
        fig.update_layout(
            title='Regime Transition Matrix',
            xaxis_title='To Regime',
            yaxis_title='From Regime'
        )
        
        return fig
        
    else:
        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            transitions,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            ax=ax
        )
        
        # Update layout
        ax.set_title('Regime Transition Matrix')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        plt.tight_layout()
        return fig

def plot_regime_weights(
    regime_history: pd.DataFrame,
    interactive: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot model weights by regime.
    
    Args:
        regime_history: DataFrame with regime history
        interactive: Whether to return Plotly or Matplotlib figure
    
    Returns:
        Plotly or Matplotlib figure
    """
    # Define weights for each regime
    weights = {
        'extreme': {'xgb': 0.2, 'lstm': 0.4, 'transformer': 0.4},
        'high': {'xgb': 0.3, 'lstm': 0.4, 'transformer': 0.3},
        'neutral': {'xgb': 0.33, 'lstm': 0.33, 'transformer': 0.34},
        'low': {'xgb': 0.4, 'lstm': 0.3, 'transformer': 0.3}
    }
    
    # Create weight DataFrame
    weight_data = []
    for regime in weights:
        for model, weight in weights[regime].items():
            weight_data.append({
                'regime': regime,
                'model': model,
                'weight': weight
            })
    
    weight_df = pd.DataFrame(weight_data)
    
    if interactive:
        # Create grouped bar chart
        fig = px.bar(
            weight_df,
            x='regime',
            y='weight',
            color='model',
            barmode='group',
            title='Model Weights by Regime'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Regime',
            yaxis_title='Weight',
            yaxis_range=[0, 0.5]
        )
        
        return fig
        
    else:
        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot grouped bars
        regimes = weight_df['regime'].unique()
        models = weight_df['model'].unique()
        x = np.arange(len(regimes))
        width = 0.25
        
        for i, model in enumerate(models):
            model_data = weight_df[weight_df['model'] == model]
            ax.bar(
                x + i * width,
                model_data['weight'],
                width,
                label=model
            )
        
        # Update layout
        ax.set_title('Model Weights by Regime')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Weight')
        ax.set_xticks(x + width)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig 