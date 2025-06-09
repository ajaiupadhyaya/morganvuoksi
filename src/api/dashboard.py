"""
API dashboard with real-time monitoring.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List
from .monitor import APIMonitor

class APIDashboard:
    """Dashboard for API monitoring and visualization."""
    
    def __init__(self, monitor: APIMonitor):
        self.monitor = monitor
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.title("API Monitoring Dashboard")
        
        # Get dashboard data
        data = self.monitor.get_api_dashboard_data()
        
        # Overview metrics
        self._render_overview_metrics(data)
        
        # API status timeline
        self._render_status_timeline(data)
        
        # API performance metrics
        self._render_performance_metrics(data)
        
        # API health status
        self._render_health_status(data)
    
    def _render_overview_metrics(self, data: Dict):
        """Render overview metrics."""
        st.header("Overview")
        
        # Calculate overall metrics
        total_apis = len(data['apis'])
        healthy_apis = sum(
            1 for api in data['apis'].values()
            if api['current_status'] and api['current_status']['is_healthy']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total APIs",
                total_apis
            )
        
        with col2:
            st.metric(
                "Healthy APIs",
                healthy_apis
            )
        
        with col3:
            st.metric(
                "Health Rate",
                f"{(healthy_apis/total_apis)*100:.1f}%" if total_apis > 0 else "N/A"
            )
    
    def _render_status_timeline(self, data: Dict):
        """Render API status timeline."""
        st.header("API Status Timeline")
        
        # Prepare timeline data
        timeline_data = []
        for api_name, api_data in data['apis'].items():
            for status in api_data['status_history']:
                timeline_data.append({
                    'timestamp': pd.to_datetime(status['timestamp']),
                    'api': api_name,
                    'status': 'Healthy' if status['is_healthy'] else 'Unhealthy',
                    'latency': status['latency']
                })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Create timeline plot
            fig = px.scatter(
                df,
                x='timestamp',
                y='api',
                color='status',
                size='latency',
                title='API Status Timeline',
                color_discrete_map={
                    'Healthy': 'green',
                    'Unhealthy': 'red'
                }
            )
            
            fig.update_layout(
                xaxis_title='Time',
                yaxis_title='API',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self, data: Dict):
        """Render API performance metrics."""
        st.header("Performance Metrics")
        
        # Prepare metrics data
        metrics_data = []
        for api_name, api_data in data['apis'].items():
            metrics = api_data['metrics']
            if metrics:
                metrics_data.append({
                    'api': api_name,
                    'uptime': metrics['uptime'] * 100,
                    'avg_latency': metrics['avg_latency'],
                    'p95_latency': metrics['p95_latency'],
                    'error_rate': metrics['error_rate'] * 100
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Create metrics plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    df,
                    x='api',
                    y='uptime',
                    title='API Uptime',
                    labels={'uptime': 'Uptime (%)'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    df,
                    x='api',
                    y='error_rate',
                    title='API Error Rate',
                    labels={'error_rate': 'Error Rate (%)'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Latency plots
            col3, col4 = st.columns(2)
            
            with col3:
                fig3 = px.bar(
                    df,
                    x='api',
                    y='avg_latency',
                    title='Average Latency',
                    labels={'avg_latency': 'Latency (s)'}
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col4:
                fig4 = px.bar(
                    df,
                    x='api',
                    y='p95_latency',
                    title='95th Percentile Latency',
                    labels={'p95_latency': 'Latency (s)'}
                )
                st.plotly_chart(fig4, use_container_width=True)
    
    def _render_health_status(self, data: Dict):
        """Render detailed API health status."""
        st.header("API Health Status")
        
        # Create health status table
        health_data = []
        for api_name, api_data in data['apis'].items():
            current_status = api_data['current_status']
            if current_status:
                health_data.append({
                    'API': api_name,
                    'Status': 'Healthy' if current_status['is_healthy'] else 'Unhealthy',
                    'Latency': f"{current_status['latency']:.3f}s",
                    'Last Check': pd.to_datetime(current_status['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        if health_data:
            df = pd.DataFrame(health_data)
            st.dataframe(
                df.style.applymap(
                    lambda x: 'background-color: green' if x == 'Healthy' else 'background-color: red',
                    subset=['Status']
                )
            )
    
    def run(self):
        """Run the dashboard."""
        st.set_page_config(
            page_title="API Monitoring Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        # Add auto-refresh
        st.empty()
        time_placeholder = st.empty()
        
        while True:
            self.render_dashboard()
            time_placeholder.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.experimental_rerun() 