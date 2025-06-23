#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal Launcher
Bloomberg-grade quantitative finance platform startup script.
"""

import os
import sys
import subprocess
import time
import signal
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MorganVuoksiTerminal:
    """Enhanced terminal launcher for Bloomberg-grade platform."""
    
    def __init__(self):
        self.processes = []
        self.root_dir = Path(__file__).parent
        self.running = False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'streamlit', 'fastapi', 'pandas', 'numpy', 'plotly', 
            'yfinance', 'scikit-learn', 'transformers'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} - Missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt"
            ], check=True)
            logger.info("✅ Dependencies installed successfully")
        else:
            logger.info("✅ All dependencies satisfied")
    
    def setup_environment(self):
        """Setup environment variables and configuration."""
        logger.info("🔧 Setting up environment...")
        
        # Create necessary directories
        directories = [
            'data', 'logs', 'outputs', 'models/saved_models',
            'outputs/reports', 'outputs/charts'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
        
        # Set environment variables
        env_vars = {
            'PYTHONPATH': str(self.root_dir),
            'MV_TERMINAL_MODE': 'production',
            'MV_LOG_LEVEL': 'INFO'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"🔧 Set {key}={value}")
    
    def start_api_server(self):
        """Start the FastAPI backend server."""
        logger.info("🚀 Starting FastAPI backend server...")
        
        api_cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.root_dir
        )
        
        self.processes.append(('FastAPI Backend', process))
        logger.info("✅ FastAPI backend started on http://localhost:8000")
        
        return process
    
    def start_streamlit_terminal(self):
        """Start the Streamlit terminal dashboard."""
        logger.info("🖥️ Starting Streamlit terminal...")
        
        terminal_cmd = [
            sys.executable, "-m", "streamlit", "run",
            "dashboard/terminal.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "dark",
            "--theme.primaryColor", "#0066cc",
            "--theme.backgroundColor", "#0a0e1a",
            "--theme.secondaryBackgroundColor", "#1e2330"
        ]
        
        process = subprocess.Popen(
            terminal_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.root_dir
        )
        
        self.processes.append(('Streamlit Terminal', process))
        logger.info("✅ Streamlit terminal started on http://localhost:8501")
        
        return process
    
    def start_next_frontend(self):
        """Start the Next.js frontend (if available)."""
        logger.info("🌐 Checking for Next.js frontend...")
        
        frontend_dir = self.root_dir / "frontend"
        if frontend_dir.exists() and (frontend_dir / "package.json").exists():
            logger.info("📦 Installing frontend dependencies...")
            
            npm_install = subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                capture_output=True,
                text=True
            )
            
            if npm_install.returncode == 0:
                logger.info("🚀 Starting Next.js frontend...")
                
                next_cmd = ["npm", "run", "dev"]
                process = subprocess.Popen(
                    next_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=frontend_dir
                )
                
                self.processes.append(('Next.js Frontend', process))
                logger.info("✅ Next.js frontend started on http://localhost:3000")
                return process
            else:
                logger.warning("⚠️ Failed to install frontend dependencies")
        else:
            logger.info("ℹ️ Next.js frontend not found, skipping...")
        
        return None
    
    def wait_for_services(self):
        """Wait for services to be ready."""
        logger.info("⏳ Waiting for services to initialize...")
        
        services = [
            ("FastAPI Backend", "http://localhost:8000/health", 30),
            ("Streamlit Terminal", "http://localhost:8501", 45)
        ]
        
        for service_name, url, timeout in services:
            logger.info(f"🔄 Checking {service_name}...")
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    import requests
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name} is ready")
                        break
                except:
                    time.sleep(2)
            else:
                logger.warning(f"⚠️ {service_name} not responding after {timeout}s")
    
    def display_dashboard(self):
        """Display the terminal dashboard information."""
        dashboard_info = """
        
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           🚀 MORGANVUOKSI ELITE TERMINAL                          ║
║                        Bloomberg-Grade Quantitative Platform                     ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  🌐 WEB INTERFACES:                                                              ║
║  • Streamlit Terminal:  http://localhost:8501                                   ║
║  • FastAPI Backend:     http://localhost:8000                                   ║
║  • Next.js Frontend:    http://localhost:3000 (if available)                    ║
║                                                                                  ║
║  📊 FEATURES AVAILABLE:                                                          ║
║  • Real-time Market Data        • AI/ML Predictions                             ║
║  • Portfolio Optimization       • Risk Management                               ║
║  • Backtesting Engine           • NLP Sentiment Analysis                        ║
║  • DCF Valuation                • RL Trading Agents                             ║
║  • Advanced Charting            • Automated Reporting                           ║
║                                                                                  ║
║  🔧 API ENDPOINTS:                                                               ║
║  • GET  /api/v1/terminal_data/{symbol}     - Real-time market data              ║
║  • POST /api/v1/predictions                - AI price predictions               ║
║  • POST /api/v1/portfolio/optimize         - Portfolio optimization             ║
║  • POST /api/v1/risk/analyze               - Risk analysis                      ║
║  • GET  /api/v1/sentiment/{symbol}         - Sentiment analysis                 ║
║  • GET  /api/v1/dcf/{symbol}               - DCF valuation                      ║
║                                                                                  ║
║  📈 TRADING FEATURES:                                                            ║
║  • Multiple ML Models (LSTM, Transformer, XGBoost, Ensemble)                    ║
║  • Portfolio Strategies (Mean-Variance, Black-Litterman, Risk Parity)           ║
║  • Risk Metrics (VaR, CVaR, Stress Testing, Position Sizing)                    ║
║  • Technical Indicators (RSI, MACD, Bollinger Bands, Moving Averages)           ║
║                                                                                  ║
║  🤖 AI CAPABILITIES:                                                             ║
║  • FinBERT Sentiment Analysis   • LSTM Price Prediction                         ║
║  • Reinforcement Learning       • Natural Language Processing                   ║
║  • Automated Report Generation  • News & Earnings Analysis                      ║
║                                                                                  ║
║  💡 GETTING STARTED:                                                             ║
║  1. Visit http://localhost:8501 for the main terminal                           ║
║  2. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)                             ║
║  3. Explore different tabs for various features                                 ║
║  4. Access API documentation at http://localhost:8000/docs                      ║
║                                                                                  ║
║  ⚠️  CONTROLS:                                                                   ║
║  • Press Ctrl+C to stop all services                                            ║
║  • Check logs in the 'logs/' directory                                          ║
║  • Reports are saved in 'outputs/' directory                                    ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

        """
        print(dashboard_info)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("🛑 Shutdown signal received...")
        self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        if not self.running:
            return
            
        self.running = False
        logger.info("🔄 Shutting down services...")
        
        for service_name, process in self.processes:
            logger.info(f"🛑 Stopping {service_name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ {service_name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ Force killing {service_name}...")
                process.kill()
                process.wait()
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
        
        logger.info("✅ All services stopped. Goodbye!")
        sys.exit(0)
    
    def run(self):
        """Main execution method."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            logger.info("🚀 Starting MorganVuoksi Elite Terminal...")
            
            # Check and install dependencies
            self.check_dependencies()
            
            # Setup environment
            self.setup_environment()
            
            # Start services
            self.start_api_server()
            time.sleep(3)  # Give API time to start
            
            self.start_streamlit_terminal()
            time.sleep(3)  # Give Streamlit time to start
            
            self.start_next_frontend()
            
            # Wait for services to be ready
            self.wait_for_services()
            
            # Display dashboard
            self.display_dashboard()
            
            self.running = True
            
            # Keep the main process alive
            logger.info("🎯 Terminal is running. Press Ctrl+C to stop.")
            while self.running:
                time.sleep(1)
                
                # Check if any process died
                for service_name, process in self.processes[:]:
                    if process.poll() is not None:
                        logger.error(f"❌ {service_name} has stopped unexpectedly")
                        self.processes.remove((service_name, process))
                
                if not self.processes:
                    logger.error("❌ All services have stopped")
                    break
        
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.shutdown()

def main():
    """Main entry point."""
    terminal = MorganVuoksiTerminal()
    terminal.run()

if __name__ == "__main__":
    main() 