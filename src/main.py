"""
Main entry point for the ML trading system.
Handles initialization, configuration, and system monitoring.
"""

import os
import sys
import logging
import yaml
from datetime import datetime
from dotenv import load_dotenv
from dash import Dash, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.regime_detector import RegimeDetector
from src.ml.learning_loop import LearningLoop
from src.visuals.regime_dashboard import RegimeDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    try:
        config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def initialize_system(config):
    """Initialize system components."""
    try:
        # Initialize regime detector
        regime_detector = RegimeDetector(
            lookback_window=config['regime_detector']['lookback_window'],
            threshold=config['regime_detector']['threshold']
        )
        
        # Initialize learning loop
        learning_loop = LearningLoop(
            model_dir=config['learning_loop']['model_dir'],
            retraining_interval=config['learning_loop']['retraining_interval'],
            performance_threshold=config['learning_loop']['performance_threshold']
        )
        
        return regime_detector, learning_loop
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        sys.exit(1)

def create_dashboard(regime_detector, learning_loop, config):
    """Create and configure the dashboard."""
    try:
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Add health check endpoint
        @app.server.route('/health')
        def health_check():
            return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
        
        # Create dashboard
        dashboard = RegimeDashboard(
            regime_history=regime_detector.get_regime_history(),
            portfolio_equity=learning_loop.get_portfolio_equity(),
            model_performance=learning_loop.get_model_performance(),
            signal_quality=learning_loop.get_signal_quality()
        )
        
        # Set up dashboard layout and callbacks
        app.layout = dashboard.create_layout()
        dashboard.setup_callbacks(app)
        
        return app
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
        sys.exit(1)

def main():
    """Main entry point for the application."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize system components
        regime_detector, learning_loop = initialize_system(config)
        logger.info("System components initialized")
        
        # Create dashboard
        app = create_dashboard(regime_detector, learning_loop, config)
        logger.info("Dashboard created successfully")
        
        # Run the dashboard
        port = int(os.getenv('DASHBOARD_PORT', 8050))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        app.run_server(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 