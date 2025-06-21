# MorganVuoksi Terminal 📈

**Bloomberg-Style Quantitative Trading Terminal**

A state-of-the-art, institutional-grade quantitative research and trading terminal designed for professional traders and quantitative analysts. MorganVuoksi Terminal combines advanced AI/ML models, comprehensive risk management, and a modern Bloomberg-style interface to deliver a complete trading platform.

## 🌟 Features

### 📊 **Market Data & Analysis**
- **Real-time Data**: Yahoo Finance, Alpaca, Polygon integration
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Interactive Charts**: Candlestick charts with volume and technical overlays
- **Market Statistics**: Comprehensive price and volume analysis

### 🤖 **AI/ML Predictions**
- **Advanced Models**: LSTM, Transformer, XGBoost, ARIMA-GARCH
- **Ensemble Learning**: Combined predictions for improved accuracy
- **Real-time Training**: Model training and validation with live data
- **Prediction Visualization**: Interactive charts with confidence intervals

### 📈 **Portfolio Optimization**
- **Multiple Strategies**: Mean-variance, Black-Litterman, Risk Parity
- **Efficient Frontier**: Interactive portfolio optimization visualization
- **Risk Management**: VaR, CVaR, stress testing, position sizing
- **Performance Analytics**: Comprehensive portfolio metrics and analysis

### 💰 **Fundamental Analysis**
- **DCF Valuation**: Discounted cash flow analysis
- **Company Metrics**: P/E, P/B, ROE, debt ratios
- **Financial Health**: Comprehensive company analysis
- **Valuation Tools**: Multiple valuation methodologies

### ⚠️ **Risk Management**
- **VaR Analysis**: Value at Risk calculations (historical, parametric, Monte Carlo)
- **Stress Testing**: Market crash, recession, volatility spike scenarios
- **Position Sizing**: Kelly Criterion and risk-based sizing
- **Risk Limits**: Automated risk limit monitoring and alerts

### 🔄 **Backtesting Engine**
- **Strategy Testing**: Multiple trading strategies
- **Performance Metrics**: Sharpe ratio, drawdown, win rate
- **Trade Analysis**: Detailed trade-by-trade analysis
- **Strategy Comparison**: Multi-strategy performance comparison

### 🎮 **Reinforcement Learning**
- **RL Agents**: TD3 and SAC algorithms for autonomous trading
- **Trading Environment**: Realistic market simulation
- **Agent Training**: Interactive training with live feedback
- **Performance Visualization**: Training progress and agent actions

### 📰 **News & Sentiment Analysis**
- **NLP Processing**: FinBERT and advanced sentiment analysis
- **News Aggregation**: Multiple news sources integration
- **Sentiment Signals**: Trading signals based on news sentiment
- **Earnings Analysis**: Earnings call transcript analysis

### 📋 **Automated Reporting**
- **Report Generation**: Automated market and portfolio reports
- **AI Summarization**: GPT-powered report summarization
- **Custom Reports**: Tailored reports for different use cases
- **Export Options**: PDF and Excel export capabilities

### 🤖 **AI Assistant**
- **LLM Integration**: GPT-powered trading assistant
- **Market Analysis**: AI-powered market interpretation
- **Strategy Guidance**: Intelligent trading strategy recommendations
- **Educational Content**: Learning resources and explanations

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Optional**: Docker, Redis (for advanced features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/morganvuoksi.git
   cd morganvuoksi
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dashboard.txt
   ```

4. **Configure API keys** (optional but recommended)
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your API keys
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   POLYGON_API_KEY=your_polygon_key
   FRED_API_KEY=your_fred_key
   OPENAI_API_KEY=your_openai_key
   NEWS_API_KEY=your_newsapi_key
   ```

5. **Launch the terminal**
   ```bash
   # Option 1: Use the startup script
   chmod +x run_terminal.sh
   ./run_terminal.sh
   
   # Option 2: Manual launch
   cd dashboard
   streamlit run terminal.py
   ```

6. **Access the terminal**
   - Open your browser and go to: `http://localhost:8501`
   - The terminal will load with a modern Bloomberg-style interface

## 📁 Configuration

### API Keys Setup

The terminal supports multiple data sources. Configure your API keys in the `.env` file:

```env
# Trading & Market Data
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
POLYGON_API_KEY=your_polygon_key

# Economic Data
FRED_API_KEY=your_fred_key

# AI & NLP
OPENAI_API_KEY=your_openai_key

# News & Sentiment
NEWS_API_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
```

### Configuration File

Edit `config/config.yaml` for advanced settings:

```yaml
# Data Sources
data_sources:
  primary: "yahoo"  # yahoo, alpaca, polygon
  backup: "alpaca"

# Risk Management
risk_management:
  max_position_size: 0.1
  max_portfolio_risk: 0.02
  stop_loss_pct: 0.05
  var_confidence_level: 0.95

# AI Models
ai_models:
  default_model: "ensemble"
  training_episodes: 100
  prediction_horizon: "5d"

# Portfolio Optimization
portfolio:
  optimization_method: "mean_variance"
  risk_tolerance: "moderate"
  rebalance_frequency: "monthly"
```

## 🎯 Usage Guide

### Market Data Tab
- **Symbol Input**: Enter any stock symbol (e.g., AAPL, TSLA, GOOGL)
- **Date Range**: Select from 1 day to 10 years
- **Technical Analysis**: View RSI, MACD, Bollinger Bands, and more
- **Interactive Charts**: Zoom, pan, and hover for detailed information

### AI Predictions Tab
- **Model Selection**: Choose from LSTM, Transformer, XGBoost, or Ensemble
- **Prediction Horizon**: Select 1 day to 30 days ahead
- **Generate Predictions**: Click to run AI models and get forecasts
- **Model Diagnostics**: View training progress and feature importance

### Portfolio Optimization Tab
- **Portfolio Symbols**: Enter multiple symbols separated by commas
- **Risk Tolerance**: Select Conservative, Moderate, or Aggressive
- **Optimization Method**: Choose from multiple optimization strategies
- **Efficient Frontier**: Visualize optimal portfolio combinations

### Risk Analysis Tab
- **Risk Metrics**: View VaR, CVaR, volatility, and drawdown
- **Stress Testing**: Run market crash and recession scenarios
- **Position Sizing**: Calculate optimal position sizes
- **Risk Limits**: Monitor portfolio risk against limits

### Backtesting Tab
- **Strategy Selection**: Choose from multiple trading strategies
- **Initial Capital**: Set starting portfolio value
- **Performance Metrics**: View Sharpe ratio, drawdown, win rate
- **Trade Analysis**: Examine individual trades and performance

### RL Simulator Tab
- **Agent Type**: Select TD3 or SAC reinforcement learning agent
- **Training Episodes**: Set number of training episodes
- **Agent Training**: Watch the agent learn and improve
- **Performance Visualization**: View training progress and actions

### News & NLP Tab
- **Sentiment Analysis**: Analyze news sentiment for trading signals
- **News Sources**: View recent news from multiple sources
- **Sentiment Distribution**: Visualize positive/negative/neutral news
- **Earnings Analysis**: Analyze earnings call transcripts

### Reports Tab
- **Report Types**: Generate market, portfolio, or risk reports
- **Time Periods**: Select from 1 week to 1 year
- **AI Summarization**: Get AI-powered report summaries
- **Export Options**: Download reports in PDF or Excel format

### LLM Assistant Tab
- **AI Chat**: Ask questions about markets, portfolios, or strategies
- **Market Analysis**: Get AI-powered market interpretation
- **Strategy Guidance**: Receive intelligent trading recommendations
- **Educational Content**: Learn about quantitative finance concepts

## 📊 Output Locations

- **Charts & Visualizations**: Displayed directly in the terminal
- **Reports**: Generated in the `outputs/` directory
- **Model Files**: Saved in `models/saved_models/`
- **Logs**: Written to `logs/` directory
- **Data Cache**: Stored in memory for faster access

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill existing Streamlit processes
   pkill -f streamlit
   # Or use a different port
   streamlit run terminal.py --server.port 8502
   ```

2. **Missing dependencies**
   ```bash
   pip install --upgrade -r requirements-dashboard.txt
   ```

3. **API key errors**
   - Check your `.env` file configuration
   - Verify API keys are valid and have sufficient credits
   - Some features work without API keys (using Yahoo Finance)

4. **Memory issues**
   - Reduce the number of symbols in portfolio optimization
   - Use shorter date ranges for large datasets
   - Restart the terminal if memory usage is high

### Performance Optimization

- **Data Caching**: The terminal caches data for faster access
- **Model Persistence**: Trained models are saved for reuse
- **Async Operations**: News fetching and API calls are asynchronous
- **Memory Management**: Large datasets are processed efficiently

## 🐳 Docker Deployment (Optional)

For production deployment or consistent environments:

```bash
# Build the Docker image
docker build -t morganvuoksi-terminal .

# Run the container
docker run -p 8501:8501 -e ALPACA_API_KEY=your_key morganvuoksi-terminal

# Or use docker-compose
docker-compose up -d
```

## 📈 Screenshots

*[Screenshots will be added here showing the terminal interface]*

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended as investment advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join our community discussions
- **Email**: Contact us at support@morganvuoksi.com

---

**MorganVuoksi Terminal** - Professional-grade quantitative trading platform for the modern trader.
