"""
Charting Functions for MorganVuoksi Terminal
All visualization functions for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Bloomberg-style color palette
BLOOMBERG_COLORS = {
    'primary': '#0066cc',
    'secondary': '#00d4aa',
    'accent': '#ff6b6b',
    'warning': '#ffa726',
    'background': '#1e2330',
    'surface': '#2a3142',
    'border': '#3a4152',
    'text_primary': '#e8eaed',
    'text_secondary': '#a0a3a9',
    'positive': '#00d4aa',
    'negative': '#ff6b6b',
    'neutral': '#a0a3a9'
}

# Bloomberg-style layout template
BLOOMBERG_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': BLOOMBERG_COLORS['background'],
    'plot_bgcolor': BLOOMBERG_COLORS['surface'],
    'font': {
        'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        'size': 12,
        'color': BLOOMBERG_COLORS['text_primary']
    },
    'margin': dict(l=50, r=50, t=50, b=50),
    'showlegend': True,
    'legend': {
        'bgcolor': BLOOMBERG_COLORS['surface'],
        'bordercolor': BLOOMBERG_COLORS['border'],
        'borderwidth': 1,
        'font': {'color': BLOOMBERG_COLORS['text_primary']}
    }
}

def create_candlestick_chart(data: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Creates an interactive candlestick chart with volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03, subplot_titles=(title, 'Volume'),
                       row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'],
                                name='OHLC',
                                increasing_line_color='#00d4aa',
                                decreasing_line_color='#ff6b6b'),
                  row=1, col=1)

    colors = ['#00d4aa' if close >= open else '#ff6b6b' for open, close in zip(data['Open'], data['Close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
                  row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark',
                      title=title, yaxis_title='Price', yaxis2_title='Volume',
                      paper_bgcolor='#1e2330', plot_bgcolor='#2a3142',
                      font=dict(color='#e8eaed'))
    return fig

def create_technical_chart(data: pd.DataFrame, title: str = "Technical Indicators") -> go.Figure:
    """Creates a chart with various technical indicators."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       subplot_titles=('RSI', 'MACD'))

    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['RSI'], 
            name='RSI',
            line=dict(color=BLOOMBERG_COLORS['primary'], width=2)
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=BLOOMBERG_COLORS['negative'], 
                     line_width=1, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=BLOOMBERG_COLORS['positive'], 
                     line_width=1, row=1, col=1)

    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MACD'], 
            name='MACD',
            line=dict(color=BLOOMBERG_COLORS['primary'], width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MACD_Signal'], 
            name='Signal Line',
            line=dict(color=BLOOMBERG_COLORS['secondary'], width=2)
        ), row=2, col=1)

    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': title,
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        height=500
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    
    return fig

def create_portfolio_chart(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    """Creates a pie chart for portfolio allocation."""
    fig = px.pie(
        values=weights.values, 
        names=weights.index, 
        title=title, 
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': title,
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        height=400
    )
    
    return fig

def create_risk_dashboard(risk_data: Dict, title="Risk Dashboard") -> go.Figure:
    """Creates a dashboard with various risk metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Volatility", "Value at Risk (VaR)", "Max Drawdown")
    )
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=risk_data.get('volatility', 0) * 100,
        title={"text": "Annual Volatility", "font": {"color": BLOOMBERG_COLORS['text_primary']}},
        number={'suffix': '%', 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        delta={'reference': 20, 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        domain={'row': 0, 'column': 0}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=risk_data.get('var_95', 0) * 100,
        title={"text": "VaR (95%)", "font": {"color": BLOOMBERG_COLORS['text_primary']}},
        number={'suffix': '%', 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        delta={'reference': -5, 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        domain={'row': 0, 'column': 1}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=risk_data.get('max_drawdown', 0) * 100,
        title={"text": "Max Drawdown", "font": {"color": BLOOMBERG_COLORS['text_primary']}},
        number={'suffix': '%', 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        delta={'reference': -10, 'font': {"color": BLOOMBERG_COLORS['text_primary']}},
        domain={'row': 0, 'column': 2}
    ), row=1, col=3)
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': title,
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        height=300
    )
    
    return fig

def create_efficient_frontier_chart(frontier_data: pd.DataFrame) -> go.Figure:
    """Creates an efficient frontier chart."""
    fig = px.scatter(
        frontier_data, 
        x='volatility', 
        y='return', 
        color='sharpe_ratio',
        hover_data={'weights': False}, 
        title="Efficient Frontier",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': "Efficient Frontier",
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        height=500
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    
    return fig

def create_prediction_chart(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Creates a chart comparing actual vs. predicted values."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual.index, 
        y=actual, 
        mode='lines', 
        name='Actual',
        line=dict(color=BLOOMBERG_COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=predicted.index, 
        y=predicted, 
        mode='lines', 
        name='Predicted', 
        line=dict(dash='dash', color=BLOOMBERG_COLORS['secondary'], width=2)
    ))
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': "AI Model Predictions",
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    
    return fig

def create_loss_curve(train_losses: List[float], test_losses: List[float]) -> go.Figure:
    """Creates a training vs. validation loss curve."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=train_losses, 
        mode='lines', 
        name='Training Loss',
        line=dict(color=BLOOMBERG_COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        y=test_losses, 
        mode='lines', 
        name='Validation Loss',
        line=dict(color=BLOOMBERG_COLORS['secondary'], width=2)
    ))
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': "Model Training Loss",
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    
    return fig

def create_feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    """Creates a bar chart for feature importance."""
    fig = px.bar(
        importance, 
        x='importance', 
        y='feature', 
        orientation='h', 
        title="Feature Importance",
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': "Feature Importance",
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor=BLOOMBERG_COLORS['border'],
        zerolinecolor=BLOOMBERG_COLORS['border'],
        showgrid=True,
        gridwidth=0.5
    )
    
    return fig

def create_sentiment_chart(sentiment_data: Dict) -> go.Figure:
    """Creates a pie chart for sentiment distribution."""
    fig = px.pie(
        values=list(sentiment_data.values()), 
        names=list(sentiment_data.keys()),
        title="News Sentiment Distribution", 
        hole=0.3,
        color_discrete_map={
            'positive': BLOOMBERG_COLORS['positive'],
            'negative': BLOOMBERG_COLORS['negative'],
            'neutral': BLOOMBERG_COLORS['neutral']
        }
    )
    
    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title={
            'text': "News Sentiment Distribution",
            'font': {'size': 16, 'color': BLOOMBERG_COLORS['text_primary']}
        },
        height=400
    )
    
    return fig 