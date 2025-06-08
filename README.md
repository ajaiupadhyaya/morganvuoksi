# ML Trading System with Regime Detection

A comprehensive machine learning trading system with regime detection, model management, and interactive visualization.

## Features

- **Regime Detection**
  - Market breadth analysis
  - Volatility term structure
  - Correlation regime detection
  - Liquidity regime monitoring
  - Composite regime classification

- **ML Models**
  - XGBoost for traditional ML
  - LSTM for sequence modeling
  - Transformer for complex patterns
  - Model ensemble with regime-based weighting

- **Interactive Dashboard**
  - Real-time regime visualization
  - Model performance tracking
  - Signal quality analysis
  - Portfolio equity overlay
  - Export capabilities (HTML, PNG)

- **Risk Management**
  - Position sizing based on regime
  - Stop-loss calculation
  - Circuit breakers
  - Performance monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-trading-system.git
cd ml-trading-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The system is configured through `config/config.yaml`. Key sections:

- `regime_detector`: Regime detection parameters
- `learning_loop`: Model training and management
- `dashboard`: Visualization settings
- `data`: Data source and storage
- `risk`: Risk management parameters

## Usage

1. Start the system:
```bash
python src/main.py
```

2. Access the dashboard:
- Open `http://localhost:8050` in your browser
- Use the interactive controls to:
  - Select time ranges
  - Toggle overlays
  - Export visualizations

3. Monitor the system:
- Check `trading_system.log` for system status
- Review model performance in the dashboard
- Monitor regime transitions

## Development

### Project Structure
```
ml-trading-system/
├── config/
│   └── config.yaml
├── src/
│   ├── ml/
│   │   ├── learning_loop.py
│   │   ├── regime_detector.py
│   │   └── safety.py
│   ├── visuals/
│   │   ├── regime_dashboard.py
│   │   └── ml_visuals.py
│   └── main.py
├── tests/
│   ├── test_learning_loop.py
│   ├── test_regime_detector.py
│   └── test_regime_dashboard.py
├── models/
├── data/
├── exports/
├── requirements.txt
└── README.md
```

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Python, Dash, and Plotly
- Uses XGBoost, TensorFlow, and PyTorch
- Inspired by academic research in regime detection and ML trading
