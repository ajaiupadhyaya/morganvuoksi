"""
Enhanced interactive dashboard with Plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class Dashboard:
    """Interactive dashboard for quantitative finance system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.theme = config.get('theme', 'plotly_white')
        self.colors = config.get('colors', px.colors.qualitative.Set1)
    
    def create_equity_curve(self, portfolio: pd.DataFrame, 
                           regimes: Optional[pd.Series] = None) -> go.Figure:
        """Create interactive equity curve with regime overlays."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Portfolio Value', 'Drawdown'))
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio['total'],
                name='Portfolio Value',
                line=dict(color=self.colors[0])
            ),
            row=1, col=1
        )
        
        # Add regime overlays if provided
        if regimes is not None:
            for regime in regimes.unique():
                regime_data = portfolio[regimes == regime]
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data['total'],
                        name=f'Regime {regime}',
                        line=dict(color=self.colors[regime % len(self.colors)])
                    ),
                    row=1, col=1
                )
        
        # Add drawdown
        drawdown = (portfolio['total'] / portfolio['total'].expanding().max() - 1)
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=drawdown,
                name='Drawdown',
                line=dict(color=self.colors[1])
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value',
            template=self.theme,
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_model_comparison(self, results: Dict[str, Dict]) -> go.Figure:
        """Create model comparison panel."""
        metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
        models = list(results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [results[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=self.colors
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            template=self.theme,
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_feature_importance(self, importance: pd.DataFrame) -> go.Figure:
        """Create feature importance plot."""
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=importance['feature'],
                y=importance['importance'],
                marker_color=self.colors
            )
        )
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Feature',
            yaxis_title='Importance',
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_signal_decay(self, signals: pd.Series, 
                           returns: pd.Series) -> go.Figure:
        """Create signal decay analysis plot."""
        # Calculate signal decay
        decay_periods = range(1, 21)  # 20 periods
        decay_correlations = []
        
        for period in decay_periods:
            correlation = signals.corr(returns.shift(period))
            decay_correlations.append(correlation)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=list(decay_periods),
                y=decay_correlations,
                mode='lines+markers',
                name='Signal Decay',
                line=dict(color=self.colors[0])
            )
        )
        
        fig.update_layout(
            title='Signal Decay Analysis',
            xaxis_title='Periods Ahead',
            yaxis_title='Correlation',
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def create_prediction_confidence(self, predictions: pd.Series, 
                                   actual: pd.Series) -> go.Figure:
        """Create prediction confidence plot."""
        # Calculate confidence intervals
        std_dev = predictions.std()
        upper_bound = predictions + 2 * std_dev
        lower_bound = predictions - 2 * std_dev
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual,
                name='Actual',
                line=dict(color=self.colors[0])
            )
        )
        
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions,
                name='Predicted',
                line=dict(color=self.colors[1])
            )
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(color=self.colors[1], width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(color=self.colors[1], width=0),
                name='95% Confidence'
            )
        )
        
        fig.update_layout(
            title='Prediction Confidence Intervals',
            xaxis_title='Date',
            yaxis_title='Value',
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def create_live_chart(self, data: pd.DataFrame, 
                         predictions: Optional[pd.Series] = None) -> go.Figure:
        """Create live price chart with prediction overlays."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Price', 'Volume'))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add predictions if provided
        if predictions is not None:
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions,
                    name='Predicted',
                    line=dict(color=self.colors[0])
                ),
                row=1, col=1
            )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=self.colors[1]
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Live Market Data',
            xaxis_title='Date',
            yaxis_title='Price',
            template=self.theme,
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_dashboard(self, data: Dict) -> None:
        """Create complete interactive dashboard."""
        st.set_page_config(page_title="Quantitative Finance Dashboard",
                          layout="wide")
        
        st.title("Quantitative Finance Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        model = st.sidebar.selectbox("Select Model", list(data['models'].keys()))
        date_range = st.sidebar.date_input("Date Range",
                                         [data['portfolio'].index[0],
                                          data['portfolio'].index[-1]])
        
        # Main dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                self.create_equity_curve(
                    data['portfolio'],
                    data.get('regimes')
                ),
                use_container_width=True
            )
            
            st.plotly_chart(
                self.create_model_comparison(data['model_results']),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.create_feature_importance(data['feature_importance']),
                use_container_width=True
            )
            
            st.plotly_chart(
                self.create_signal_decay(
                    data['signals'],
                    data['returns']
                ),
                use_container_width=True
            )
        
        # Live data section
        st.header("Live Market Data")
        st.plotly_chart(
            self.create_live_chart(
                data['market_data'],
                data.get('predictions')
            ),
            use_container_width=True
        )
        
        # Export options
        st.sidebar.header("Export")
        if st.sidebar.button("Export Dashboard"):
            # Implement export functionality
            pass 
