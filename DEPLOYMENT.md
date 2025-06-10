# MorganVuoksi Deployment Guide

This guide provides detailed instructions for deploying the MorganVuoksi quantitative trading platform.

## Prerequisites

### System Requirements
- macOS 24.5.0 or later
- Python 3.11 or later
- 16GB RAM minimum (32GB recommended)
- 100GB free disk space
- Stable internet connection

### Required Accounts & API Keys
1. **Market Data Providers**
   - Bloomberg Terminal (BLPAPI)
   - Refinitiv Eikon/LSEG
   - Interactive Brokers TWS
   - Polygon.io Professional
   - IEX Cloud Professional
   - Quandl/Nasdaq Data Link Premium
   - S&P Capital IQ
   - CRSP/Compustat
   - FactSet

2. **Alternative Data**
   - RavenPack
   - FRED API
   - SEC EDGAR API
   - Twitter API
   - Reddit API
   - Google Trends API

3. **Trading Brokers**
   - Interactive Brokers
   - Alpaca

## Step-by-Step Installation

### 1. System Setup

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system packages
brew install python@3.11
brew install redis
brew install kafka
brew install influxdb
brew install cmake
brew install boost
brew install zeromq
```

### 2. Project Setup

```bash
# Clone repository
git clone https://github.com/yourusername/morganvuoksi.git
cd morganvuoksi

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Bloomberg API
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi

# Install Interactive Brokers API
pip install ib_insync
```

### 3. Configuration

1. **API Credentials**
   Create `.env` file in project root:
   ```bash
   # Market Data
   BLOOMBERG_API_KEY=your_bloomberg_key
   REFINITIV_API_KEY=your_refinitiv_key
   POLYGON_API_KEY=your_polygon_key
   IEX_API_KEY=your_iex_key
   QUANDL_API_KEY=your_quandl_key
   CAPITAL_IQ_API_KEY=your_capital_iq_key
   CRSP_API_KEY=your_crsp_key
   FACTSET_API_KEY=your_factset_key

   # Alternative Data
   RAVENPACK_API_KEY=your_ravenpack_key
   FRED_API_KEY=your_fred_key
   TWITTER_API_KEY=your_twitter_key
   REDDIT_API_KEY=your_reddit_key
   GOOGLE_TRENDS_API_KEY=your_google_trends_key

   # Trading
   IB_ACCOUNT=your_ib_account
   IB_PASSWORD=your_ib_password
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   ```

2. **Database Configuration**
   Edit `config/database.yaml`:
   ```yaml
   redis:
     host: localhost
     port: 6379
     password: your_redis_password

   kafka:
     bootstrap_servers: localhost:9092
     topic_prefix: morganvuoksi

   influxdb:
     url: http://localhost:8086
     token: your_influxdb_token
     org: your_org
     bucket: market_data
   ```

3. **Model Configuration**
   Edit `config/models.yaml`:
   ```yaml
   lstm:
     hidden_size: 64
     num_layers: 2
     dropout: 0.2
     batch_size: 32
     epochs: 50
     learning_rate: 0.001

   transformer:
     d_model: 512
     nhead: 8
     num_layers: 6
     dim_feedforward: 2048
     dropout: 0.1

   ppo:
     learning_rate: 0.0003
     gamma: 0.99
     eps_clip: 0.2
     K_epochs: 10
   ```

### 4. Data Infrastructure Setup

1. **Start Redis**
   ```bash
   brew services start redis
   ```

2. **Start Kafka**
   ```bash
   brew services start kafka
   ```

3. **Start InfluxDB**
   ```bash
   brew services start influxdb
   ```

4. **Initialize Databases**
   ```bash
   # Create InfluxDB bucket
   influx bucket create -n market_data

   # Create Kafka topics
   kafka-topics --create --topic market_data --bootstrap-server localhost:9092
   kafka-topics --create --topic orders --bootstrap-server localhost:9092
   kafka-topics --create --topic signals --bootstrap-server localhost:9092
   ```

### 5. Model Training

1. **Download Historical Data**
   ```bash
   python scripts/download_data.py --start 2020-01-01 --end 2024-03-14
   ```

2. **Train Models**
   ```bash
   # Train all models
   python run_all.py --train

   # Train specific models
   python run.py --model lstm --train
   python run.py --model transformer --train
   python run.py --model ppo --train
   ```

### 6. Start Services

1. **Start Data Pipeline**
   ```bash
   python src/data/pipeline.py
   ```

2. **Start ML Models**
   ```bash
   python src/ml/ecosystem.py
   ```

3. **Start Trading System**
   ```bash
   python src/trading/infrastructure.py
   ```

4. **Start Monitoring**
   ```bash
   # Start Prometheus
   prometheus --config.file=prometheus.yml

   # Start Grafana
   brew services start grafana
   ```

### 7. Verify Installation

1. **Check Services**
   ```bash
   # Check Redis
   redis-cli ping  # Should return PONG

   # Check Kafka
   kafka-topics --list --bootstrap-server localhost:9092

   # Check InfluxDB
   influx ping
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Check Logs**
   ```bash
   tail -f logs/app.log
   ```

## Monitoring & Maintenance

### Daily Checks
1. Monitor system logs:
   ```bash
   tail -f logs/app.log
   ```

2. Check model performance:
   ```bash
   python scripts/check_performance.py
   ```

3. Monitor trading activity:
   ```bash
   python scripts/monitor_trading.py
   ```

### Weekly Maintenance
1. Update market data:
   ```bash
   python scripts/update_data.py
   ```

2. Retrain models:
   ```bash
   python run_all.py --retrain
   ```

3. Backup databases:
   ```bash
   ./scripts/backup.sh
   ```

## Troubleshooting

### Common Issues

1. **Bloomberg API Connection**
   ```bash
   # Check Bloomberg service
   blpapi-version
   
   # Restart Bloomberg service
   brew services restart bloomberg
   ```

2. **Redis Connection**
   ```bash
   # Check Redis status
   brew services list | grep redis
   
   # Restart Redis
   brew services restart redis
   ```

3. **Kafka Issues**
   ```bash
   # Check Kafka status
   brew services list | grep kafka
   
   # Restart Kafka
   brew services restart kafka
   ```

4. **Model Training Issues**
   ```bash
   # Clear model cache
   rm -rf models/cache/*
   
   # Retrain with debug logging
   python run_all.py --train --debug
   ```

### Performance Optimization

1. **Memory Usage**
   ```bash
   # Monitor memory usage
   top -o mem
   
   # Clear Redis cache if needed
   redis-cli FLUSHALL
   ```

2. **CPU Usage**
   ```bash
   # Monitor CPU usage
   top -o cpu
   
   # Adjust number of workers
   export NUM_WORKERS=4
   ```

3. **Disk Space**
   ```bash
   # Check disk usage
   df -h
   
   # Clean old data
   python scripts/cleanup.py --days 30
   ```

## Security

1. **API Key Rotation**
   ```bash
   # Rotate API keys
   python scripts/rotate_keys.py
   ```

2. **Access Control**
   ```bash
   # Update permissions
   chmod 600 .env
   chmod 600 config/*.yaml
   ```

3. **Audit Logs**
   ```bash
   # Check audit logs
   tail -f logs/audit.log
   ```

## Support

For technical support:
1. Check the documentation in `docs/`
2. Review system logs in `logs/`
3. Contact support at support@morganvuoksi.com

## License

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited. 