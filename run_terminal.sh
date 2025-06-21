#!/bin/bash

# MorganVuoksi Terminal Startup Script
# This script sets up and runs the Bloomberg-style quantitative trading terminal

echo "🚀 Starting MorganVuoksi Terminal..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements-dashboard.txt

# Check if config file exists
if [ ! -f "config/config.yaml" ]; then
    echo "⚠️  Warning: config/config.yaml not found. Using default configuration."
fi

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

# Run the terminal
echo "🎯 Launching MorganVuoksi Terminal..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the terminal"
echo ""

streamlit run dashboard/terminal.py 