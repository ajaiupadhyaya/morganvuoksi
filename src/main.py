"""
Main application file with API monitoring and dashboard.
"""
import yaml
import streamlit as st
from api.monitor import APIMonitor
from api.dashboard import APIDashboard
from utils.logging import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def main():
    """Main application entry point."""
    # Load configuration
    config = load_config('config.yaml')
    
    if not config:
        st.error("Failed to load configuration. Please check config.yaml file.")
        return
    
    # Initialize API monitor
    monitor = APIMonitor(config)
    
    # Initialize dashboard
    dashboard = APIDashboard(monitor)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run dashboard
    dashboard.run()

if __name__ == "__main__":
    main() 