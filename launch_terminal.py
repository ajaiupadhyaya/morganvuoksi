#!/usr/bin/env python3
"""
MorganVuoksi Terminal Launcher
Simple script to launch the Bloomberg-style quantitative trading terminal.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the MorganVuoksi Terminal."""
    
    print("🚀 MorganVuoksi Terminal Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard/terminal.py").exists():
        print("❌ Error: dashboard/terminal.py not found!")
        print("Please run this script from the morganvuoksi root directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas", "numpy", "yfinance"])
        print("✅ Dependencies installed")
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    print("\n🎯 Launching MorganVuoksi Terminal...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the terminal")
    print("=" * 50)
    
    try:
        # Launch the terminal
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/terminal.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 MorganVuoksi Terminal stopped.")
    except Exception as e:
        print(f"❌ Error launching terminal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 