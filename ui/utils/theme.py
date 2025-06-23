"""
Bloomberg-Style Theme Manager
Professional dark theme with Bloomberg Terminal color specifications.
"""

import streamlit as st
from typing import Dict, Optional

class BloombergTheme:
    """Bloomberg Terminal style theme manager."""
    
    # Color Palette
    COLORS = {
        # Base Theme
        'primary': '#0a0a0a',           # Primary background
        'secondary': '#1a1a1a',         # Secondary panels
        'tertiary': '#252525',          # Tertiary panels
        'text_primary': '#ffffff',       # Primary text
        'text_secondary': '#808080',     # Secondary text
        
        # Data-Driven Colors
        'gains': '#00ff00',             # Positive changes
        'losses': '#ff0000',            # Negative changes
        'warnings': '#ffff00',          # Warning states
        'headers': '#00bfff',           # Header text
        'alerts': '#ff8c00',            # Alert notifications
        
        # Bloomberg Accent Colors
        'bloomberg_blue': '#0066cc',    # Bloomberg signature blue
        'bloomberg_orange': '#ff8c42',  # Bloomberg orange
        'terminal_green': '#00d4aa',    # Terminal green
        'border': '#333333',            # Border color
        'hover': '#2a2a2a',            # Hover states
    }
    
    # Font Specifications
    FONTS = {
        'monospace': '"Roboto Mono", "Consolas", "Monaco", monospace',
        'primary': '"Helvetica Neue", "Arial", sans-serif',
        'secondary': '"Inter", sans-serif'
    }
    
    @classmethod
    def get_css(cls) -> str:
        """Generate Bloomberg-style CSS for Streamlit."""
        return f"""
        <style>
            /* Global Styles */
            .stApp {{
                background-color: {cls.COLORS['primary']};
                color: {cls.COLORS['text_primary']};
                font-family: {cls.FONTS['primary']};
            }}
            
            /* Main Container */
            .main {{
                background-color: {cls.COLORS['primary']};
                padding: 0;
                margin: 0;
            }}
            
            /* Sidebar Styling */
            .stSidebar {{
                background-color: {cls.COLORS['secondary']};
                border-right: 1px solid {cls.COLORS['border']};
            }}
            
            .stSidebar .sidebar-content {{
                background-color: {cls.COLORS['secondary']};
                padding: 1rem;
            }}
            
            /* Tab Styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0;
                background-color: {cls.COLORS['secondary']};
                border-bottom: 1px solid {cls.COLORS['border']};
                padding: 0;
                margin: 0;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: {cls.COLORS['tertiary']};
                color: {cls.COLORS['text_secondary']};
                border: none;
                border-right: 1px solid {cls.COLORS['border']};
                padding: 12px 20px;
                font-weight: 500;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.2s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {cls.COLORS['bloomberg_blue']};
                color: {cls.COLORS['text_primary']};
                font-weight: 600;
            }}
            
            .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
                background-color: {cls.COLORS['hover']};
                color: {cls.COLORS['text_primary']};
            }}
            
            /* Metric Cards */
            .metric-card {{
                background: linear-gradient(135deg, {cls.COLORS['secondary']} 0%, {cls.COLORS['tertiary']} 100%);
                border: 1px solid {cls.COLORS['border']};
                border-radius: 4px;
                padding: 12px;
                margin: 4px 0;
                transition: all 0.2s ease;
            }}
            
            .metric-card:hover {{
                border-color: {cls.COLORS['bloomberg_blue']};
                transform: translateY(-1px);
            }}
            
            .metric-card .label {{
                color: {cls.COLORS['text_secondary']};
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
                font-family: {cls.FONTS['monospace']};
            }}
            
            .metric-card .value {{
                color: {cls.COLORS['text_primary']};
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 2px;
                font-family: {cls.FONTS['monospace']};
            }}
            
            .metric-card .change {{
                font-size: 12px;
                font-weight: 500;
                font-family: {cls.FONTS['monospace']};
            }}
            
            /* Data-driven colors */
            .positive {{
                color: {cls.COLORS['gains']} !important;
            }}
            
            .negative {{
                color: {cls.COLORS['losses']} !important;
            }}
            
            .warning {{
                color: {cls.COLORS['warnings']} !important;
            }}
            
            .header-text {{
                color: {cls.COLORS['headers']} !important;
            }}
            
            .alert-text {{
                color: {cls.COLORS['alerts']} !important;
            }}
            
            /* Button Styling */
            .stButton > button {{
                background: linear-gradient(135deg, {cls.COLORS['bloomberg_blue']} 0%, #0052a3 100%);
                color: {cls.COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.2s ease;
                font-family: {cls.FONTS['monospace']};
            }}
            
            .stButton > button:hover {{
                background: linear-gradient(135deg, #0052a3 0%, #003d7a 100%);
                transform: translateY(-1px);
            }}
            
            /* Input Styling */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div > select {{
                background-color: {cls.COLORS['tertiary']};
                border: 1px solid {cls.COLORS['border']};
                border-radius: 4px;
                color: {cls.COLORS['text_primary']};
                font-size: 12px;
                padding: 6px 8px;
                font-family: {cls.FONTS['monospace']};
            }}
            
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus,
            .stSelectbox > div > div > select:focus {{
                border-color: {cls.COLORS['bloomberg_blue']};
                box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
            }}
            
            /* Data Tables */
            .dataframe {{
                background-color: {cls.COLORS['secondary']};
                border: 1px solid {cls.COLORS['border']};
                border-radius: 4px;
                font-family: {cls.FONTS['monospace']};
                font-size: 11px;
            }}
            
            .dataframe th {{
                background-color: {cls.COLORS['tertiary']};
                color: {cls.COLORS['headers']};
                font-weight: 600;
                padding: 8px;
                border-bottom: 1px solid {cls.COLORS['border']};
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 10px;
            }}
            
            .dataframe td {{
                padding: 6px 8px;
                border-bottom: 1px solid {cls.COLORS['border']};
                color: {cls.COLORS['text_primary']};
            }}
            
            .dataframe tr:hover {{
                background-color: {cls.COLORS['hover']};
            }}
            
            /* Status Indicators */
            .status-indicator {{
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 6px;
            }}
            
            .status-live {{
                background-color: {cls.COLORS['gains']};
                box-shadow: 0 0 8px rgba(0, 255, 0, 0.5);
                animation: pulse 2s infinite;
            }}
            
            .status-warning {{
                background-color: {cls.COLORS['warnings']};
                box-shadow: 0 0 8px rgba(255, 255, 0, 0.5);
            }}
            
            .status-error {{
                background-color: {cls.COLORS['losses']};
                box-shadow: 0 0 8px rgba(255, 0, 0, 0.5);
            }}
            
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            /* Container Styling */
            .terminal-container {{
                background-color: {cls.COLORS['secondary']};
                border: 1px solid {cls.COLORS['border']};
                border-radius: 4px;
                padding: 12px;
                margin: 8px 0;
            }}
            
            .terminal-header {{
                background-color: {cls.COLORS['tertiary']};
                border-bottom: 1px solid {cls.COLORS['border']};
                padding: 8px 12px;
                margin: -12px -12px 12px -12px;
                border-radius: 4px 4px 0 0;
            }}
            
            .terminal-header h3 {{
                color: {cls.COLORS['headers']};
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin: 0;
                font-family: {cls.FONTS['monospace']};
            }}
            
            /* Chart Containers */
            .chart-container {{
                background-color: {cls.COLORS['secondary']};
                border: 1px solid {cls.COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                margin: 8px 0;
            }}
            
            /* Scrollbar Styling */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: {cls.COLORS['secondary']};
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {cls.COLORS['border']};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {cls.COLORS['bloomberg_blue']};
            }}
            
            /* Hide Streamlit Elements */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
            
            /* Compact spacing */
            .element-container {{
                margin: 0 !important;
                padding: 0 !important;
            }}
            
            .stMarkdown {{
                margin: 0 !important;
                padding: 0 !important;
            }}
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {{
                .metric-card {{
                    padding: 8px;
                }}
                
                .metric-card .value {{
                    font-size: 16px;
                }}
                
                .stTabs [data-baseweb="tab"] {{
                    padding: 8px 12px;
                    font-size: 12px;
                }}
            }}
        </style>
        """
    
    @classmethod
    def apply_theme(cls):
        """Apply Bloomberg theme to Streamlit app."""
        st.markdown(cls.get_css(), unsafe_allow_html=True)
    
    @classmethod
    def create_metric_card(cls, label: str, value: str, change: Optional[str] = None, 
                          change_type: str = 'neutral') -> str:
        """Create a Bloomberg-style metric card."""
        change_class = {
            'positive': 'positive',
            'negative': 'negative', 
            'neutral': '',
            'warning': 'warning'
        }.get(change_type, '')
        
        change_html = f'<div class="change {change_class}">{change}</div>' if change else ''
        
        return f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {change_html}
        </div>
        """
    
    @classmethod
    def create_header(cls, title: str, status: str = 'live') -> str:
        """Create a Bloomberg-style panel header."""
        status_class = f'status-{status}'
        
        return f"""
        <div class="terminal-header">
            <h3>
                <span class="status-indicator {status_class}"></span>
                {title}
            </h3>
        </div>
        """
    
    @classmethod
    def format_number(cls, value: float, precision: int = 2, 
                     show_sign: bool = True, percentage: bool = False) -> str:
        """Format numbers with Bloomberg-style formatting."""
        if percentage:
            formatted = f"{value:.{precision}f}%"
        else:
            formatted = f"{value:,.{precision}f}"
        
        if show_sign and value > 0:
            formatted = f"+{formatted}"
        
        return formatted
    
    @classmethod
    def get_color_for_value(cls, value: float, threshold: float = 0) -> str:
        """Get color class for a value based on positive/negative."""
        if value > threshold:
            return 'positive'
        elif value < threshold:
            return 'negative'
        else:
            return 'neutral'