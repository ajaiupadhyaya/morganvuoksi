# MorganVuoksi Quant Terminal

A comprehensive Bloomberg-style quantitative trading and research platform built with Streamlit. This institutional-grade terminal provides real-time market data, AI/ML predictions, portfolio optimization, risk management, and trading simulation capabilities.

![MorganVuoksi Terminal](https://via.placeholder.com/800x400/1f77b4/ffffff?text=MorganVuoksi+Terminal+Screenshot)

## 🎯 Features

- **📈 Market Data Viewer**: Real-time data, technical indicators, interactive charts
- **🤖 AI/ML Predictions**: LSTM, XGBoost, Transformer, ARIMA-GARCH models
- **⚙️ Backtesting Engine**: Strategy testing, performance metrics, trade analysis
- **📊 Portfolio Optimizer**: Mean-variance, risk parity, efficient frontier
- **🧠 NLP & Sentiment**: Market sentiment, news analysis, social media scoring
- **📉 Valuation Tools**: DCF, comparable analysis, LBO modeling
- **💸 Trade Simulator**: Execution simulation, market impact, algorithmic trading
- **🧾 Report Generator**: Automated reports, performance analysis, exports
- **🧪 Risk Management**: VaR analysis, drawdown tracking, risk alerts
- **🧬 LLM Assistant**: AI-powered analysis, strategy recommendations

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB free space
- **Internet**: Required for market data and API access

### Required Software
- **Python Package Manager**: pip (included with Python)
- **Git**: For cloning the repository
- **C++ Compiler**: Required for some ML dependencies (optional, for GPU acceleration)

### Optional Dependencies
- **Docker**: For containerized deployment
- **Redis**: For caching and session management
- **PostgreSQL**: For persistent data storage

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd morganvuoksi
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements-dashboard.txt

# Install additional development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Verify Installation
```bash
# Check Python version
python --version  # Should be 3.8+

# Check Streamlit installation
streamlit --version

# Test imports
python -c "import streamlit, plotly, pandas, numpy, yfinance; print('✅ All dependencies installed successfully')"
```

## 🔧 Configuration

### 1. Environment Variables

Create a `.env` file in the root directory:

```bash
# Create .env file
touch .env
```

Add the following environment variables to `.env`:

```env
# =============================================================================
# REQUIRED APIs (Free Tier Available)
# =============================================================================

# Yahoo Finance (Free - no API key required)
# No configuration needed - uses yfinance library

# OpenAI GPT (Optional - for LLM Assistant)
OPENAI_API_KEY=your_openai_api_key_here

# =============================================================================
# RECOMMENDED APIs (Production Use)
# =============================================================================

# Alpaca Trading (Free paper trading available)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading for testing

# Polygon.io (Real-time market data)
POLYGON_API_KEY=your_polygon_api_key_here

# =============================================================================
# OPTIONAL APIs (Advanced Features)
# =============================================================================

# Hugging Face (Pre-trained ML models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Weights & Biases (Experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here

# =============================================================================
# DATABASE & CACHING (Optional)
# =============================================================================

# PostgreSQL (For persistent data)
DATABASE_URL=postgresql://username:password@localhost:5432/morganvuoksi

# Redis (For caching and sessions)
REDIS_URL=redis://localhost:6379

# =============================================================================
# LOGGING & MONITORING
# =============================================================================

# Logging level
LOG_LEVEL=INFO

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

### 2. Configuration File

Edit `config/config.yaml` to customize system behavior:

```yaml
# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
models:
  lstm:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    batch_size: 32
    epochs: 50
    learning_rate: 0.001
    sequence_length: 10
  
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    subsample: 0.8
    colsample_bytree: 0.8
  
  transformer:
    d_model: 64
    n_heads: 4
    n_layers: 2
    dropout: 0.1
    batch_size: 32
    epochs: 50
    learning_rate: 0.001
  
  arima_garch:
    max_p: 5
    max_d: 2
    max_q: 5
    seasonal: true
    m: 12
    garch_p: 1
    garch_q: 1

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
risk:
  max_position_size: 0.05  # 5% max position size
  stop_loss: 0.02          # 2% stop loss
  max_drawdown: 0.15       # 15% max drawdown
  var_confidence: 0.95     # 95% VaR confidence level
  circuit_breakers:
    max_drawdown: 0.1      # 10% circuit breaker
    volatility_threshold: 0.3

# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================
backtesting:
  initial_capital: 100000   # $100,000 starting capital
  commission: 0.001         # 0.1% commission
  slippage: 0.0005          # 0.05% slippage
  rebalance_frequency: "daily"
  default_symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
data:
  update_frequency: "1min"  # Data update frequency
  max_retries: 3            # API retry attempts
  cache_duration: 300       # Cache duration in seconds
  default_period: "1y"      # Default time period
  default_symbols: ["AAPL", "MSFT", "GOOGL"]

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
dashboard:
  port: 8501
  debug: false
  theme: "dark"
  refresh_interval: 60      # Auto-refresh interval in seconds
  export:
    enabled: true
    formats: ["html", "png", "pdf"]
    directory: "reports"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging:
  level: "INFO"
  file: "logs/morganvuoksi.log"
  max_size: 10485760        # 10MB
  backup_count: 5
```

## 🚀 Running the Terminal

### Method 1: Direct Launch (Recommended)
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Launch the terminal
streamlit run dashboard/terminal.py
```

### Method 2: Using Python Launcher
```bash
python launch_terminal.py
```

### Method 3: Using Shell Script
```bash
# Make script executable (macOS/Linux only)
chmod +x run_terminal.sh

# Run the script
./run_terminal.sh
```

### Method 4: Demo Version (For Testing)
```bash
streamlit run demo_terminal.py
```

### Method 5: Custom Port
```bash
streamlit run dashboard/terminal.py --server.port 8502
```

## 📊 Usage Guide

### Accessing the Terminal
1. Open your web browser
2. Navigate to `http://localhost:8501`
3. The terminal will load with the dark Bloomberg-style interface

### Sidebar Configuration
- **Symbol**: Enter stock symbol (e.g., "AAPL", "MSFT")
- **Time Period**: Select data timeframe (1mo to 5y)
- **Strategy**: Choose trading strategy
- **ML Models**: Select which models to train
- **Risk Parameters**: Configure position sizing and stop losses

### Tab Overview

#### 📈 Market Data
- View real-time and historical price data
- Technical indicators (RSI, Moving Averages, Volume)
- Interactive charts with zoom and pan
- Data table with recent prices

#### 🤖 AI/ML Predictions
- Click "Train Models & Generate Predictions"
- Select models (LSTM, XGBoost, Transformer, ARIMA-GARCH)
- View prediction charts and performance metrics
- Feature importance analysis

#### ⚙️ Backtesting
- Configure backtest parameters (capital, commission, slippage)
- Click "Run Backtest" to execute strategy
- View equity curve and performance metrics
- Analyze trade-by-trade results

#### 📊 Portfolio Optimizer
- Enter multiple symbols (comma-separated)
- Select optimization method (Mean-Variance, Risk Parity, etc.)
- View optimized weights and efficient frontier
- Portfolio allocation pie chart

#### 🧠 NLP & Sentiment
- Market sentiment timeline
- News volume analysis
- Social media sentiment scoring
- Sentiment metrics dashboard

#### 📉 Valuation Tools
- DCF model with customizable parameters
- Comparable company analysis
- LBO modeling scenarios
- Dividend discount models

#### 💸 Trade Simulator
- Configure trade parameters (symbol, size, price)
- Select order type and algorithm
- View execution results and timeline
- Market impact analysis

#### 🧾 Report Generator
- Select report type and date range
- Configure export format (PDF, HTML, Excel)
- Generate automated reports
- View performance metrics

#### 🧪 Risk Management
- VaR analysis and distribution
- Portfolio drawdown tracking
- Risk alerts and notifications
- Correlation analysis

#### 🧬 LLM Assistant
- Natural language queries about markets
- AI-powered strategy recommendations
- Risk assessment and analysis
- Market commentary generation

## 📁 Output Locations

### Reports and Exports
- **Reports**: `reports/` directory
- **Charts**: Exported as PNG/PDF to `reports/`
- **Data**: Processed data saved to `data/processed/`

### Logs and Debugging
- **Application Logs**: `logs/morganvuoksi.log`
- **Error Logs**: `logs/errors.log`
- **Performance Logs**: `logs/performance.log`

### Model Artifacts
- **Trained Models**: `models/saved_models/`
- **Model Metrics**: `models/metrics/`
- **Feature Importance**: `models/features/`

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'streamlit'
pip install streamlit plotly pandas numpy yfinance

# Error: ModuleNotFoundError for other packages
pip install -r requirements-dashboard.txt
```

#### 2. API Key Issues
```bash
# Error: API key not found
# Solution: Check .env file exists and contains correct API keys
cat .env

# Error: API rate limit exceeded
# Solution: Wait and retry, or upgrade API plan
```

#### 3. Port Already in Use
```bash
# Error: Port 8501 is already in use
# Solution: Use different port
streamlit run dashboard/terminal.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

#### 4. Memory Issues
```bash
# Error: Out of memory during model training
# Solution: Reduce batch size in config/config.yaml
models:
  lstm:
    batch_size: 16  # Reduce from 32
```

#### 5. Data Loading Issues
```bash
# Error: Unable to fetch market data
# Solution: Check internet connection and API status
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['regularMarketPrice'])"
```

### Debugging Tips

#### 1. Enable Debug Mode
```yaml
# In config/config.yaml
dashboard:
  debug: true
```

#### 2. Check Logs
```bash
# View application logs
tail -f logs/morganvuoksi.log

# View error logs
tail -f logs/errors.log
```

#### 3. Reset State
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache/

# Clear model cache
rm -rf models/saved_models/
```

#### 4. Verify Configuration
```bash
# Test configuration loading
python -c "from src.config import Config; config = Config(); print('Config loaded successfully')"
```

## 🐳 Docker Deployment

### Using Docker Compose
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d
```

### Using Docker Directly
```bash
# Build the image
docker build -t morganvuoksi-terminal .

# Run the container
docker run -p 8501:8501 \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  morganvuoksi-terminal
```

### Production Deployment
```bash
# Build production image
docker build -f Dockerfile.prod -t morganvuoksi-terminal:prod .

# Run with production settings
docker run -d \
  -p 8501:8501 \
  --name morganvuoksi-prod \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  morganvuoksi-terminal:prod
```

## 🔑 API Key Setup

### Required APIs (Free Tier)

#### Yahoo Finance
- **Status**: Free, no API key required
- **Rate Limits**: 2,000 requests per hour
- **Usage**: Market data, historical prices

#### OpenAI GPT (Optional)
- **Cost**: $0.002 per 1K tokens
- **Setup**: Get API key from [OpenAI Platform](https://platform.openai.com/)
- **Usage**: LLM Assistant functionality

### Recommended APIs (Production)

#### Alpaca Trading
- **Cost**: Free (paper trading), $0.01 per share (live)
- **Setup**: Get API keys from [Alpaca Markets](https://alpaca.markets/)
- **Usage**: Commission-free trading, market data

#### Polygon.io
- **Cost**: $99/month (Starter plan)
- **Setup**: Get API key from [Polygon.io](https://polygon.io/)
- **Usage**: Real-time market data, options data

### Optional APIs (Advanced)

#### Hugging Face
- **Cost**: Free tier available
- **Setup**: Get API key from [Hugging Face](https://huggingface.co/)
- **Usage**: Pre-trained ML models

#### Weights & Biases
- **Cost**: Free tier available
- **Setup**: Get API key from [Weights & Biases](https://wandb.ai/)
- **Usage**: Experiment tracking, model versioning

## 📈 Performance Optimization

### Caching
- Streamlit automatically caches expensive computations
- Market data cached for 5 minutes by default
- Model predictions cached for 1 hour

### GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CuPy for GPU-accelerated NumPy
pip install cupy-cuda11x
```

### Memory Optimization
```yaml
# In config/config.yaml
models:
  lstm:
    batch_size: 16  # Reduce for lower memory usage
    sequence_length: 5  # Reduce sequence length
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests
pytest tests/test_models.py

# Integration tests
pytest tests/test_integration.py

# Performance tests
pytest tests/test_performance.py
```

### Test Coverage
```bash
pytest --cov=src tests/
```

## 📞 Support

### Getting Help
1. **Check Documentation**: Review this README and `TERMINAL_GUIDE.md`
2. **View Logs**: Check `logs/` directory for error messages
3. **GitHub Issues**: Open an issue on the project repository
4. **Community**: Join the project Discord/Slack for community support

### Common Questions

**Q: How do I add new ML models?**
A: Create your model in `src/models/` following the existing pattern, then add it to the model selection in `dashboard/terminal.py`.

**Q: Can I use my own data sources?**
A: Yes, modify `src/data/pipeline.py` to integrate your data sources.

**Q: How do I deploy to production?**
A: Use Docker for containerized deployment or follow the cloud deployment instructions in `DEPLOYMENT.md`.

**Q: Is this suitable for live trading?**
A: The terminal includes paper trading capabilities. For live trading, ensure proper risk management and regulatory compliance.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Roadmap

### Phase 2: Advanced Features
- [ ] Real-time data streaming
- [ ] Advanced NLP models
- [ ] Reinforcement learning agents
- [ ] Multi-asset portfolio optimization

### Phase 3: Enterprise Features
- [ ] Multi-user support
- [ ] Advanced security
- [ ] Cloud deployment
- [ ] API endpoints

### Phase 4: AI Enhancement
- [ ] Advanced LLM integration
- [ ] Automated strategy generation
- [ ] Predictive analytics
- [ ] Natural language trading

---

**MorganVuoksi Terminal v1.0** - Powered by Advanced Quantitative Analytics

For questions and support, please refer to the documentation or open an issue on GitHub.
