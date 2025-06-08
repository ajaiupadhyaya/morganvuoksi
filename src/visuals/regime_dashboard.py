"""
Regime visualization dashboard for monitoring market regimes and model performance.
Provides both interactive (Plotly) and static (Matplotlib) visualizations.
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
import logging
from pathlib import Path
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeDashboard:
    """Dashboard for visualizing regime detection and model performance."""
    
    def __init__(
        self,
        regime_history: pd.DataFrame,
        portfolio_equity: Optional[pd.Series] = None,
        model_performance: Optional[Dict] = None,
        signal_quality: Optional[Dict] = None
    ):
        """
        Initialize dashboard with data.
        
        Args:
            regime_history: DataFrame with regime history
            portfolio_equity: Optional portfolio equity curve
            model_performance: Optional dict with model performance metrics
            signal_quality: Optional dict with signal quality metrics
        """
        self.regime_history = regime_history
        self.portfolio_equity = portfolio_equity
        self.model_performance = model_performance or {}
        self.signal_quality = signal_quality or {}
        
        # Set style parameters
        self.style = {
            'font_family': 'Arial',
            'title_font_size': 16,
            'axis_font_size': 12,
            'legend_font_size': 10,
            'color_palette': px.colors.qualitative.Set3,
            'background_color': 'white',
            'grid_color': 'lightgray',
            'template': 'plotly_white',
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': 'Arial'
            }
        }
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Set up layout
        self.app.layout = self._create_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _create_layout(self) -> html.Div:
        """Create dashboard layout."""
        return html.Div([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(
                        "Regime Detection Dashboard",
                        className="text-center my-4"
                    )
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Time Range"),
                        dbc.CardBody([
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date=self.regime_history['timestamp'].min(),
                                end_date=self.regime_history['timestamp'].max(),
                                display_format='YYYY-MM-DD'
                            )
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Display Options"),
                        dbc.CardBody([
                            dbc.Checklist(
                                id='display-options',
                                options=[
                                    {'label': 'Show Portfolio Equity', 'value': 'equity'},
                                    {'label': 'Show Model Performance', 'value': 'performance'},
                                    {'label': 'Show Signal Quality', 'value': 'quality'}
                                ],
                                value=['equity'],
                                switch=True
                            )
                        ])
                    ])
                ], width=3),
                
                # Main content
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='core-indicators-plot')
                        ], label='Core Indicators'),
                        
                        dbc.Tab([
                            dcc.Graph(id='composite-regime-plot')
                        ], label='Composite Regime'),
                        
                        dbc.Tab([
                            dcc.Graph(id='model-performance-plot')
                        ], label='Model Performance'),
                        
                        dbc.Tab([
                            dcc.Graph(id='signal-quality-plot')
                        ], label='Signal Quality')
                    ])
                ], width=9)
            ])
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            Output('core-indicators-plot', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_core_indicators(start_date, end_date):
            mask = (self.regime_history['timestamp'] >= start_date) & \
                   (self.regime_history['timestamp'] <= end_date)
            return self.plot_core_indicators(
                regime_history=self.regime_history[mask],
                interactive=True
            )
        
        @self.app.callback(
            Output('composite-regime-plot', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('display-options', 'value')]
        )
        def update_composite_regime(start_date, end_date, display_options):
            mask = (self.regime_history['timestamp'] >= start_date) & \
                   (self.regime_history['timestamp'] <= end_date)
            return self.plot_composite_regime(
                regime_history=self.regime_history[mask],
                portfolio_equity=self.portfolio_equity if 'equity' in display_options else None,
                interactive=True
            )
        
        @self.app.callback(
            Output('model-performance-plot', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_model_performance(start_date, end_date):
            return self.plot_model_performance(interactive=True)
        
        @self.app.callback(
            Output('signal-quality-plot', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_signal_quality(start_date, end_date):
            return self.plot_signal_quality(interactive=True)
    
    def plot_core_indicators(
        self,
        regime_history: Optional[pd.DataFrame] = None,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot core regime indicators.
        
        Args:
            regime_history: Optional DataFrame with regime history
            interactive: Whether to return Plotly or Matplotlib figure
            save_path: Optional path to save figure
        
        Returns:
            Plotly or Matplotlib figure
        """
        regime_history = regime_history or self.regime_history
        
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=(
                    'Market Breadth',
                    'IV/RV Spread',
                    'Correlation Regime',
                    'Liquidity Indicators'
                ),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Add traces for each indicator
            indicators = ['breadth', 'volatility', 'correlation', 'liquidity']
            colors = self.style['color_palette'][:4]
            
            for i, (indicator, color) in enumerate(zip(indicators, colors), 1):
                # Add value line
                fig.add_trace(
                    go.Scatter(
                        x=regime_history['timestamp'],
                        y=regime_history[f'{indicator}_value'],
                        name=f'{indicator.title()} Value',
                        line=dict(color=color),
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
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
                            ),
                            hovertemplate='%{x}<br>%{y:.2f}<br>%{fullData.name}<extra></extra>'
                        ),
                        row=i,
                        col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                title_text='Core Regime Indicators',
                template=self.style['template'],
                hovermode=self.style['hovermode'],
                hoverlabel=self.style['hoverlabel'],
                font=dict(
                    family=self.style['font_family'],
                    size=self.style['title_font_size']
                ),
                plot_bgcolor=self.style['background_color']
            )
            
            # Update axes
            for i in range(1, 5):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        else:
            # Create Matplotlib figure
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
            fig.suptitle('Core Regime Indicators', fontsize=self.style['title_font_size'])
            
            # Plot each indicator
            indicators = ['breadth', 'volatility', 'correlation', 'liquidity']
            colors = self.style['color_palette'][:4]
            
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
                
                ax.set_title(f'{indicator.title()} Indicator', fontsize=self.style['axis_font_size'])
                ax.legend(fontsize=self.style['legend_font_size'])
                ax.grid(True, color=self.style['grid_color'])
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_composite_regime(
        self,
        regime_history: Optional[pd.DataFrame] = None,
        portfolio_equity: Optional[pd.Series] = None,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot composite regime with overlays.
        
        Args:
            regime_history: Optional DataFrame with regime history
            portfolio_equity: Optional portfolio equity curve
            interactive: Whether to return Plotly or Matplotlib figure
            save_path: Optional path to save figure
        
        Returns:
            Plotly or Matplotlib figure
        """
        regime_history = regime_history or self.regime_history
        portfolio_equity = portfolio_equity or self.portfolio_equity
        
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
                        ),
                        hovertemplate='%{x}<br>%{y}<extra></extra>'
                    )
                )
            
            # Add portfolio equity overlay if available
            if portfolio_equity is not None:
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_equity.index,
                        y=portfolio_equity,
                        name='Portfolio Equity',
                        line=dict(color='black', width=1),
                        yaxis='y2',
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title='Composite Regime with Portfolio Equity Overlay',
                template=self.style['template'],
                hovermode=self.style['hovermode'],
                hoverlabel=self.style['hoverlabel'],
                yaxis=dict(
                    title='Regime',
                    categoryorder='array',
                    categoryarray=['extreme', 'high', 'neutral', 'low', 'unknown']
                ),
                yaxis2=dict(
                    title='Portfolio Equity',
                    overlaying='y',
                    side='right'
                ),
                showlegend=True,
                font=dict(
                    family=self.style['font_family'],
                    size=self.style['title_font_size']
                ),
                plot_bgcolor=self.style['background_color']
            )
            
            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.style['grid_color']
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.style['grid_color']
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
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
            
            # Add portfolio equity overlay if available
            if portfolio_equity is not None:
                ax2 = ax1.twinx()
                ax2.plot(
                    portfolio_equity.index,
                    portfolio_equity,
                    'k-',
                    label='Portfolio Equity',
                    alpha=0.5
                )
                ax2.set_ylabel('Portfolio Equity')
            
            # Update layout
            ax1.set_title('Composite Regime with Portfolio Equity Overlay', fontsize=self.style['title_font_size'])
            ax1.set_ylabel('Regime', fontsize=self.style['axis_font_size'])
            ax1.legend(fontsize=self.style['legend_font_size'])
            ax1.grid(True, color=self.style['grid_color'])
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_model_performance(
        self,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot model performance metrics by regime.
        
        Args:
            interactive: Whether to return Plotly or Matplotlib figure
            save_path: Optional path to save figure
        
        Returns:
            Plotly or Matplotlib figure
        """
        if not self.model_performance:
            logger.warning("No model performance data available")
            return None
        
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    'Sharpe Ratio by Regime',
                    'Precision by Regime',
                    'Returns by Regime'
                ),
                vertical_spacing=0.1
            )
            
            # Plot each metric
            metrics = ['sharpe', 'precision', 'returns']
            for i, metric in enumerate(metrics, 1):
                data = []
                for regime, perf in self.model_performance.items():
                    data.append({
                        'regime': regime,
                        'value': perf[metric],
                        'model': 'All Models'
                    })
                
                df = pd.DataFrame(data)
                
                fig.add_trace(
                    go.Bar(
                        x=df['regime'],
                        y=df['value'],
                        name=metric.title(),
                        marker_color=self.style['color_palette'][i-1],
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    ),
                    row=i,
                    col=1
                )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text='Model Performance by Regime',
                template=self.style['template'],
                hovermode=self.style['hovermode'],
                hoverlabel=self.style['hoverlabel'],
                font=dict(
                    family=self.style['font_family'],
                    size=self.style['title_font_size']
                ),
                plot_bgcolor=self.style['background_color']
            )
            
            # Update axes
            for i in range(1, 4):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        else:
            # Create Matplotlib figure
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            fig.suptitle('Model Performance by Regime', fontsize=self.style['title_font_size'])
            
            # Plot each metric
            metrics = ['sharpe', 'precision', 'returns']
            for i, (metric, ax) in enumerate(zip(metrics, axes)):
                data = []
                for regime, perf in self.model_performance.items():
                    data.append({
                        'regime': regime,
                        'value': perf[metric]
                    })
                
                df = pd.DataFrame(data)
                
                ax.bar(
                    df['regime'],
                    df['value'],
                    color=self.style['color_palette'][i]
                )
                ax.set_title(f'{metric.title()} by Regime', fontsize=self.style['axis_font_size'])
                ax.grid(True, color=self.style['grid_color'])
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_signal_quality(
        self,
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot signal quality metrics by regime.
        
        Args:
            interactive: Whether to return Plotly or Matplotlib figure
            save_path: Optional path to save figure
        
        Returns:
            Plotly or Matplotlib figure
        """
        if not self.signal_quality:
            logger.warning("No signal quality data available")
            return None
        
        if interactive:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    'Signal Decay by Regime',
                    'Signal Quality Metrics'
                ),
                vertical_spacing=0.1
            )
            
            # Plot signal decay
            regimes = list(self.signal_quality.keys())
            for regime in regimes:
                decay = self.signal_quality[regime]['decay']
                fig.add_trace(
                    go.Scatter(
                        x=decay.index,
                        y=decay.values,
                        name=f'{regime} Decay',
                        line=dict(color=self.style['color_palette'][regimes.index(regime)]),
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    ),
                    row=1,
                    col=1
                )
            
            # Plot quality metrics
            metrics = ['precision', 'recall', 'f1']
            for i, metric in enumerate(metrics):
                data = []
                for regime in regimes:
                    data.append({
                        'regime': regime,
                        'value': self.signal_quality[regime][metric]
                    })
                
                df = pd.DataFrame(data)
                
                fig.add_trace(
                    go.Bar(
                        x=df['regime'],
                        y=df['value'],
                        name=metric.title(),
                        marker_color=self.style['color_palette'][i],
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    ),
                    row=2,
                    col=1
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text='Signal Quality by Regime',
                template=self.style['template'],
                hovermode=self.style['hovermode'],
                hoverlabel=self.style['hoverlabel'],
                font=dict(
                    family=self.style['font_family'],
                    size=self.style['title_font_size']
                ),
                plot_bgcolor=self.style['background_color']
            )
            
            # Update axes
            for i in range(1, 3):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=self.style['grid_color'],
                    row=i,
                    col=1
                )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        else:
            # Create Matplotlib figure
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle('Signal Quality by Regime', fontsize=self.style['title_font_size'])
            
            # Plot signal decay
            regimes = list(self.signal_quality.keys())
            for regime in regimes:
                decay = self.signal_quality[regime]['decay']
                axes[0].plot(
                    decay.index,
                    decay.values,
                    label=f'{regime} Decay',
                    color=self.style['color_palette'][regimes.index(regime)]
                )
            
            axes[0].set_title('Signal Decay by Regime', fontsize=self.style['axis_font_size'])
            axes[0].legend(fontsize=self.style['legend_font_size'])
            axes[0].grid(True, color=self.style['grid_color'])
            
            # Plot quality metrics
            metrics = ['precision', 'recall', 'f1']
            x = np.arange(len(regimes))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                data = []
                for regime in regimes:
                    data.append(self.signal_quality[regime][metric])
                
                axes[1].bar(
                    x + i * width,
                    data,
                    width,
                    label=metric.title(),
                    color=self.style['color_palette'][i]
                )
            
            axes[1].set_title('Signal Quality Metrics', fontsize=self.style['axis_font_size'])
            axes[1].set_xticks(x + width)
            axes[1].set_xticklabels(regimes)
            axes[1].legend(fontsize=self.style['legend_font_size'])
            axes[1].grid(True, color=self.style['grid_color'])
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def create_dashboard(
        self,
        output_dir: str,
        interactive: bool = True
    ) -> None:
        """
        Create complete dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save dashboard files
            interactive: Whether to create interactive or static visualizations
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_core_indicators(
            interactive=interactive,
            save_path=str(output_path / 'core_indicators.html' if interactive else 'core_indicators.png')
        )
        
        self.plot_composite_regime(
            interactive=interactive,
            save_path=str(output_path / 'composite_regime.html' if interactive else 'composite_regime.png')
        )
        
        self.plot_model_performance(
            interactive=interactive,
            save_path=str(output_path / 'model_performance.html' if interactive else 'model_performance.png')
        )
        
        self.plot_signal_quality(
            interactive=interactive,
            save_path=str(output_path / 'signal_quality.html' if interactive else 'signal_quality.png')
        )
        
        logger.info(f"Dashboard created in {output_dir}")
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        self.app.run_server(debug=debug, port=port) 