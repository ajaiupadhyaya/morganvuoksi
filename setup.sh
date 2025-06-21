#!/bin/bash

# MorganVuoksi Terminal - Quick Setup Script
# This script sets up the terminal for deployment

set -e

echo "🚀 MorganVuoksi Terminal Setup"
echo "================================"

# Make deployment script executable
chmod +x deploy.sh
chmod +x run_terminal.sh

# Create necessary directories
mkdir -p logs outputs models/saved_models

# Check if .env exists, create if not
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# MorganVuoksi Terminal - Environment Configuration
# Fill in your API keys below

# Trading & Market Data
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
POLYGON_API_KEY=your_polygon_api_key_here

# Economic Data
FRED_API_KEY=your_fred_api_key_here

# AI & NLP
OPENAI_API_KEY=your_openai_api_key_here

# News & Sentiment
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here
EOF
    echo "✅ Created .env template. Please edit it with your API keys."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: ./deploy.sh local     (for local deployment)"
echo "3. Run: ./deploy.sh docker    (for Docker deployment)"
echo "4. Run: ./deploy.sh cloud     (for cloud deployment)"
echo ""
echo "For help: ./deploy.sh help" 