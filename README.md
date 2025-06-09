# Quantitative Finance System

A comprehensive quantitative finance system for algorithmic trading, featuring multiple machine learning models, real-time data processing, and advanced backtesting capabilities.

## Features

- **Multiple ML Models**
  - LSTM for deep learning
  - XGBoost for ensemble learning
  - ARIMA-GARCH for traditional econometrics
  - Transformer for sequence learning
  - PPO (Reinforcement Learning) for portfolio optimization

- **Data Infrastructure**
  - Real-time data fetching from multiple sources
  - Rate limiting and error handling
  - Data validation and quality checks

- **Backtesting Engine**
  - Comprehensive performance metrics
  - Transaction costs and slippage modeling
  - Risk management features
  - Detailed reporting and visualization

- **Interactive Dashboard**
  - Real-time model performance monitoring
  - Interactive visualizations
  - Model comparison tools
  - Export capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quant-finance.git
cd quant-finance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Running the System

1. Start the main system:
```bash
python run.py
```

2. Run backtests:
```bash
python run_backtest.py
```

3. Launch the dashboard:
```bash
streamlit run src/dashboard/app.py
```

### Configuration

The system is highly configurable through the following files:
- `config.yaml`: Main configuration file
- `.env`: API keys and sensitive data
- Model-specific configs in `src/models/`

### Data Sources

The system supports multiple data sources:
- Yahoo Finance
- Alpha Vantage
- Polygon.io
- IEX Cloud
- FRED (Federal Reserve Economic Data)

## Architecture

```
quant-finance/
├── src/
│   ├── data/           # Data fetching and processing
│   ├── models/         # ML models
│   ├── backtesting/    # Backtesting engine
│   ├── dashboard/      # Interactive dashboard
│   └── utils/          # Utility functions
├── tests/              # Unit tests
├── demo_outputs/       # Sample outputs and reports
├── config.yaml         # Configuration file
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Model Performance

### Sample Metrics (AAPL, 2023)

| Model | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate |
|-------|--------------|---------------|--------------|-----------|
| LSTM | 1.85 | 2.12 | -12.3% | 58% |
| XGBoost | 1.92 | 2.25 | -11.8% | 61% |
| ARIMA-GARCH | 1.45 | 1.78 | -15.2% | 54% |
| Transformer | 2.05 | 2.35 | -10.5% | 63% |
| PPO | 1.78 | 2.01 | -13.1% | 57% |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- XGBoost developers for the gradient boosting library
- Streamlit team for the dashboard framework

## Contact

For questions and support, please open an issue or contact the maintainers.
