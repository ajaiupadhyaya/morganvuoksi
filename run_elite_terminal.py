#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal - Main Runner
Launch the complete Bloomberg-grade quantitative finance terminal.
"""

import subprocess
import sys
import os
import time
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api_server():
    """Start the FastAPI backend server."""
    try:
        logger.info("🚀 Starting FastAPI backend server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")

def run_streamlit_terminal():
    """Start the Streamlit terminal interface."""
    try:
        logger.info("🖥️ Starting Streamlit terminal...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "terminal_elite.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit terminal: {e}")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'fastapi', 'uvicorn', 'pandas', 'numpy', 
        'plotly', 'yfinance', 'scikit-learn', 'torch', 'xgboost'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"⚠️ Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"] + missing_packages
            )
            logger.info("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def print_banner():
    """Print the terminal startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               🚀 MORGANVUOKSI ELITE TERMINAL 🚀               ║
    ║                                                              ║
    ║            Bloomberg-Grade Quantitative Finance Platform     ║
    ║                                                              ║
    ║  🔥 Real-time Market Data    📊 Portfolio Optimization       ║
    ║  🤖 AI/ML Predictions        ⚠️  Advanced Risk Management     ║
    ║  🎯 Algorithmic Trading      � Technical Analysis           ║
    ║  💰 DCF Valuation           📰 NLP Sentiment Analysis        ║
    ║  🔄 Backtesting Engine       🎮 RL Trading Agents            ║
    ║                                                              ║
    ║  Access Points:                                              ║
    ║  • Terminal UI:  http://localhost:8501                       ║
    ║  • API Backend:  http://localhost:8000                       ║
    ║  • API Docs:     http://localhost:8000/docs                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point for the elite terminal."""
    print_banner()
    
    logger.info("🔍 Checking system requirements...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install required packages.")
        return
    
    logger.info("✅ System requirements satisfied")
    
    try:
        # Start API server in background thread
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        # Wait a moment for API to start
        time.sleep(3)
        
        logger.info("🌐 API server starting in background...")
        logger.info("📊 Launching Bloomberg-style terminal interface...")
        
        # Start Streamlit terminal (this will be the main process)
        run_streamlit_terminal()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down MorganVuoksi Elite Terminal...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()