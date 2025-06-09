"""
Main entry point for the ML trading system.
Handles initialization, configuration, and system monitoring.
"""

import os
import sys
import logging
import yaml
import json
import time
import signal
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dash import Dash, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# Configure API error logging
api_logger = logging.getLogger('api_errors')
api_handler = logging.FileHandler('logs/api_errors.log')
api_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
api_logger.addHandler(api_handler)

# Configure model performance logging
model_logger = logging.getLogger('model_performance')
model_handler = logging.FileHandler('logs/model_performance.log')
model_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
model_logger.addHandler(model_handler)

# System state
system_state = {
    'start_time': datetime.now(),
    'last_health_check': datetime.now(),
    'consecutive_errors': 0,
    'circuit_breaker_triggered': False,
    'last_signal_time': None,
    'last_api_check': None,
    'error_counts': {
        'api': 0,
        'model': 0,
        'signal': 0
    }
}

class ConfigFileHandler(FileSystemEventHandler):
    """Watchdog handler for config file changes."""
    def on_modified(self, event):
        if event.src_path.endswith('config.yaml'):
            logger.info("Configuration file changed, reloading...")
            try:
                config = load_config()
                # Update system components with new config
                regime_detector.update_config(config['regime_detector'])
                learning_loop.update_config(config['learning_loop'])
                logger.info("Configuration reloaded successfully")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")

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

def check_system_health():
    """Check system health and trigger circuit breaker if needed."""
    try:
        current_time = datetime.now()
        
        # Check for API connectivity
        api_status = check_api_connectivity()
        if not api_status:
            system_state['error_counts']['api'] += 1
            logger.warning("API connectivity check failed")
        
        # Check model performance
        model_status = check_model_performance()
        if not model_status:
            system_state['error_counts']['model'] += 1
            logger.warning("Model performance check failed")
        
        # Check signal generation
        signal_status = check_signal_generation()
        if not signal_status:
            system_state['error_counts']['signal'] += 1
            logger.warning("Signal generation check failed")
        
        # Update system state
        system_state['last_health_check'] = current_time
        
        # Check for circuit breaker conditions
        if (system_state['error_counts']['api'] > 5 or
            system_state['error_counts']['model'] > 3 or
            system_state['error_counts']['signal'] > 3):
            trigger_circuit_breaker()
        
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def check_api_connectivity():
    """Check connectivity to external APIs."""
    try:
        # Add your API connectivity checks here
        return True
    except Exception as e:
        logger.error(f"API connectivity check failed: {e}")
        return False

def check_model_performance():
    """Check model performance metrics."""
    try:
        # Add your model performance checks here
        return True
    except Exception as e:
        logger.error(f"Model performance check failed: {e}")
        return False

def check_signal_generation():
    """Check signal generation status."""
    try:
        # Add your signal generation checks here
        return True
    except Exception as e:
        logger.error(f"Signal generation check failed: {e}")
        return False

def trigger_circuit_breaker():
    """Trigger circuit breaker and notify administrators."""
    system_state['circuit_breaker_triggered'] = True
    logger.critical("Circuit breaker triggered - system entering safe mode")
    
    # Send alert to Slack (if configured)
    if os.getenv('SLACK_WEBHOOK_URL'):
        try:
            message = {
                "text": "🚨 Circuit breaker triggered in ML Trading System",
                "attachments": [{
                    "color": "danger",
                    "fields": [
                        {"title": "API Errors", "value": system_state['error_counts']['api']},
                        {"title": "Model Errors", "value": system_state['error_counts']['model']},
                        {"title": "Signal Errors", "value": system_state['error_counts']['signal']}
                    ]
                }]
            }
            requests.post(os.getenv('SLACK_WEBHOOK_URL'), json=message)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

def create_dashboard(regime_detector, learning_loop, config):
    """Create and configure the dashboard."""
    try:
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Add health check endpoint
        @app.server.route('/health')
        def health_check():
            return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
        
        # Add system status endpoint
        @app.server.route('/status')
        def system_status():
            try:
                status = {
                    'status': 'operational',
                    'timestamp': datetime.now().isoformat(),
                    'uptime': str(datetime.now() - system_state['start_time']),
                    'components': {
                        'regime_detector': {
                            'status': 'operational',
                            'current_regime': regime_detector.get_current_regime(),
                            'regime_history': len(regime_detector.get_regime_history())
                        },
                        'learning_loop': {
                            'status': 'operational',
                            'model_performance': learning_loop.get_model_performance(),
                            'last_retraining': learning_loop.get_last_retraining_time()
                        },
                        'data_ingestion': {
                            'status': 'operational',
                            'last_update': learning_loop.get_last_data_update()
                        }
                    },
                    'system_metrics': {
                        'uptime': get_system_uptime(),
                        'memory_usage': get_memory_usage(),
                        'cpu_usage': get_cpu_usage()
                    },
                    'error_counts': system_state['error_counts'],
                    'circuit_breaker': system_state['circuit_breaker_triggered']
                }
                return json.dumps(status, indent=2)
            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                return {'status': 'error', 'message': str(e)}, 500
        
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

def get_system_uptime():
    """Get system uptime in seconds."""
    try:
        with open('/proc/uptime', 'r') as f:
            return float(f.read().split()[0])
    except:
        return 0

def get_memory_usage():
    """Get system memory usage."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except:
        return 0

def get_cpu_usage():
    """Get system CPU usage."""
    try:
        import psutil
        return psutil.cpu_percent()
    except:
        return 0

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info("Received shutdown signal, initiating graceful shutdown...")
    # Add cleanup code here
    sys.exit(0)

def main():
    """Main entry point for the application."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Set up config file watcher
        event_handler = ConfigFileHandler()
        observer = Observer()
        observer.schedule(event_handler, path='config', recursive=False)
        observer.start()
        
        # Initialize system components
        regime_detector, learning_loop = initialize_system(config)
        logger.info("System components initialized")
        
        # Start health check thread
        def health_check_thread():
            while True:
                check_system_health()
                time.sleep(60)  # Check every minute
        
        health_thread = threading.Thread(target=health_check_thread, daemon=True)
        health_thread.start()
        
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
    finally:
        observer.stop()
        observer.join()

if __name__ == '__main__':
    main() 