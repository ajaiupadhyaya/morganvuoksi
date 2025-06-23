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

def create_candlestick_chart(data: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Creates an interactive candlestick chart with volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03, subplot_titles=(title, 'Volume'),
                       row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'],
                                name='OHLC',
                                increasing_line_color='#00ff88',
                                decreasing_line_color='#ff4444'),
                  row=1, col=1)

    colors = ['#00ff88' if close >= open else '#ff4444' for open, close in zip(data['Open'], data['Close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
                  row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark',
                      title=title, yaxis_title='Price', yaxis2_title='Volume')
    return fig

def create_technical_chart(data: pd.DataFrame, title: str = "Technical Indicators") -> go.Figure:
    """Creates a chart with various technical indicators."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       subplot_titles=('RSI', 'MACD'))

    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal Line'), row=2, col=1)

    fig.update_layout(template='plotly_dark', title=title, showlegend=False)
    return fig

def create_portfolio_chart(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    """Creates a pie chart for portfolio allocation."""
    fig = px.pie(values=weights.values, names=weights.index, title=title, hole=0.3)
    fig.update_layout(template='plotly_dark')
    return fig

def create_risk_dashboard(risk_data: Dict, title="Risk Dashboard") -> go.Figure:
    """Creates a dashboard with various risk metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Volatility", "Value at Risk (VaR)", "Max Drawdown")
    )
    fig.add_trace(go.Indicator(
        mode="number",
        value=risk_data.get('volatility', 0) * 100,
        title={"text": "Annual Volatility"},
        number={'suffix': '%'}),
        row=1, col=1
    )
    fig.add_trace(go.Indicator(
        mode="number",
        value=risk_data.get('var_95', 0) * 100,
        title={"text": "VaR (95%)"},
        number={'suffix': '%'}),
        row=1, col=2
    )
    fig.add_trace(go.Indicator(
        mode="number",
        value=risk_data.get('max_drawdown', 0) * 100,
        title={"text": "Max Drawdown"},
        number={'suffix': '%'}),
        row=1, col=3
    )
    fig.update_layout(template='plotly_dark', title=title)
    return fig

def create_efficient_frontier_chart(frontier_data: pd.DataFrame) -> go.Figure:
    """Creates an efficient frontier chart."""
    fig = px.scatter(frontier_data, x='volatility', y='return', color='sharpe_ratio',
                     hover_data={'weights': False}, title="Efficient Frontier")
    fig.update_layout(template='plotly_dark')
    return fig

def create_prediction_chart(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Creates a chart comparing actual vs. predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=predicted.index, y=predicted, mode='lines', name='Predicted', line=dict(dash='dash')))
    fig.update_layout(template='plotly_dark', title="AI Model Predictions")
    return fig

def create_loss_curve(train_losses: List[float], test_losses: List[float]) -> go.Figure:
    """Creates a training vs. validation loss curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=test_losses, mode='lines', name='Validation Loss'))
    fig.update_layout(template='plotly_dark', title="Model Training Loss")
    return fig

def create_feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    """Creates a bar chart for feature importance."""
    fig = px.bar(importance, x='importance', y='feature', orientation='h', title="Feature Importance")
    fig.update_layout(template='plotly_dark')
    return fig

def create_sentiment_chart(sentiment_data: Dict) -> go.Figure:
    """Creates a pie chart for sentiment distribution."""
    fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                 title="News Sentiment Distribution", hole=0.3)
    fig.update_layout(template='plotly_dark')
    return fig 