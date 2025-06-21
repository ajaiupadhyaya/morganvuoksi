#!/bin/bash

# MorganVuoksi Terminal Startup Script
# Bloomberg-style quantitative trading terminal

echo "🚀 Starting MorganVuoksi Terminal..."
echo "📈 Bloomberg-style Quantitative Trading Terminal"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dashboard.txt

# Check if required API keys are set
echo "🔑 Checking API configuration..."
if [ -z "$ALPACA_API_KEY" ] && [ -z "$POLYGON_API_KEY" ]; then
    echo "⚠️  Warning: No API keys found. Some features may be limited."
    echo "   Set ALPACA_API_KEY, POLYGON_API_KEY, or other API keys for full functionality."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the terminal
echo "🎯 Launching MorganVuoksi Terminal..."
echo "🌐 Terminal will be available at: http://localhost:8501"
echo "📱 Press Ctrl+C to stop the terminal"
echo ""

# Run the terminal
cd dashboard
streamlit run terminal.py --server.port 8501 --server.address 0.0.0.0 --server.headless true 