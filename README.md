# ML Trading System with Regime Detection

A real-time trading system that uses machine learning to detect market regimes and generate trading signals.

## Features

- Real-time market data integration (Alpaca, Yahoo Finance, Polygon)
- ML-based regime detection
- Adaptive model weighting
- Interactive dashboard
- System health monitoring
- Docker and Heroku deployment support

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Ingestion │────▶│ Regime Detection│────▶│  ML Inference   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Signal Generation│────▶│ Position Sizing │────▶│  Visualization  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

- Python 3.9+
- Docker (optional)
- Heroku CLI (for cloud deployment)
- API keys for data providers

## Quick Start

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-trading-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. Run the system:
```bash
./run.sh
```

### Docker Deployment

1. Build and run with Docker:
```bash
docker-compose up --build
```

### Heroku Deployment

1. Install Heroku CLI and login:
```bash
heroku login
```

2. Create Heroku app:
```bash
heroku create your-app-name
```

3. Set environment variables:
```bash
heroku config:set ALPACA_API_KEY=your_key
heroku config:set ALPACA_API_SECRET=your_secret
heroku config:set POLYGON_API_KEY=your_key
```

4. Deploy:
```bash
git push heroku main
```

## System Monitoring

The system includes several monitoring endpoints:

- Health Check: `http://localhost:8050/health`
- System Status: `http://localhost:8050/status`
- Dashboard: `http://localhost:8050`

Logs are stored in:
- `logs/trading_system.log` - System logs
- `logs/api_errors.log` - API error logs
- `logs/model_performance.log` - Model performance metrics

## Configuration

### Environment Variables

Required environment variables:
- `ALPACA_API_KEY`: Your Alpaca API key
- `ALPACA_API_SECRET`: Your Alpaca API secret
- `POLYGON_API_KEY`: Your Polygon API key
- `DEBUG`: Set to 'True' for development
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Configuration File

The system uses `config/config.yaml` for:
- Regime detection parameters
- Model training settings
- Dashboard configuration
- Risk management rules

## Architecture Details

### Data Flow

1. **Data Ingestion**
   - Real-time market data from APIs
   - Historical data for model training
   - Data validation and preprocessing

2. **Regime Detection**
   - Multiple regime indicators
   - Adaptive thresholds
   - Regime history tracking

3. **ML Inference**
   - Model ensemble management
   - Real-time predictions
   - Performance monitoring

4. **Signal Generation**
   - Signal quality assessment
   - Position sizing
   - Risk management

5. **Visualization**
   - Interactive dashboard
   - Real-time updates
   - Performance metrics

### Model Lifecycle

1. **Training**
   - Initial model training
   - Periodic retraining
   - Performance validation

2. **Inference**
   - Real-time predictions
   - Model weighting
   - Signal generation

3. **Monitoring**
   - Performance tracking
   - Drift detection
   - Error logging

## Troubleshooting

Common issues and solutions:

1. **API Connection Issues**
   - Verify API keys in .env
   - Check API rate limits
   - Monitor API error logs

2. **Model Performance**
   - Check model performance logs
   - Verify data quality
   - Monitor regime detection

3. **Dashboard Issues**
   - Clear browser cache
   - Check server logs
   - Verify port availability

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
