# API Monitoring Dashboard

A real-time API monitoring dashboard built with Streamlit that tracks API health, performance metrics, and provides visualizations.

## Features

- Real-time API health monitoring
- Performance metrics tracking (uptime, latency, error rates)
- Interactive visualizations
- Historical data analysis
- Configurable monitoring intervals
- Redis-based data storage

## Prerequisites

- macOS (tested on macOS 24.5.0)
- Python 3.8+
- Redis server
- API keys for monitored services

## Detailed Installation Guide

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python and Redis
```bash
# Install Python
brew install python@3.11

# Install Redis
brew install redis

# Start Redis service
brew services start redis

# Verify Redis is running
redis-cli ping  # Should return PONG
```

### 3. Clone and Setup Project
```bash
# Clone the repository
git clone <repository-url>
cd api-monitoring-dashboard

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. API Key Setup

#### Weather API (OpenWeatherMap)
1. Go to https://openweathermap.org/
2. Sign up for a free account
3. Go to "My API Keys" section
4. Copy your API key

#### News API
1. Go to https://newsapi.org/
2. Sign up for a free account
3. Go to "API Keys" section
4. Copy your API key

#### GitHub API
- No API key required for basic status checks
- For higher rate limits, create a personal access token:
  1. Go to GitHub Settings > Developer Settings > Personal Access Tokens
  2. Generate new token with `repo` scope
  3. Copy the token

### 5. Environment Configuration

Create a `.env` file in the project root:
```bash
touch .env
```

Add your API keys to `.env`:
```
WEATHER_API_KEY=your_weather_api_key_here
NEWS_API_KEY=your_news_api_key_here
GITHUB_TOKEN=your_github_token_here  # Optional
```

### 6. Configure Monitoring

Edit `config.yaml` to customize your monitoring setup:
```yaml
# Example configuration
apis:
  weather_api:
    endpoint: https://api.openweathermap.org/data/2.5/weather
    headers:
      appid: ${WEATHER_API_KEY}
    params:
      q: "London"  # Default city to check
      units: "metric"
  
  news_api:
    endpoint: https://newsapi.org/v2/top-headlines
    headers:
      X-Api-Key: ${NEWS_API_KEY}
    params:
      country: "us"  # Default country for news
  
  github_api:
    endpoint: https://api.github.com/status
    headers:
      Accept: application/vnd.github.v3+json
      Authorization: token ${GITHUB_TOKEN}  # Optional
```

### 7. Running the Application

1. Ensure Redis is running:
```bash
# Check Redis status
brew services list | grep redis

# If not running, start it
brew services start redis
```

2. Start the dashboard:
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the dashboard
streamlit run src/main.py
```

3. Access the dashboard:
- Open your browser
- Go to http://localhost:8501

### 8. Troubleshooting

#### Redis Connection Issues
```bash
# Check Redis logs
tail -f /usr/local/var/log/redis.log

# Restart Redis if needed
brew services restart redis
```

#### Port Conflicts
If port 8501 is in use:
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>
```

#### API Rate Limits
- Free tier limits:
  - OpenWeatherMap: 60 calls/minute
  - News API: 100 calls/day
  - GitHub: 60 calls/hour (without token)
- Adjust `check_interval` in `config.yaml` accordingly

### 9. Maintenance

#### Updating Dependencies
```bash
# Update pip
pip install --upgrade pip

# Update requirements
pip install -r requirements.txt --upgrade
```

#### Clearing Redis Data
```bash
# Connect to Redis CLI
redis-cli

# Clear all data
FLUSHALL

# Exit Redis CLI
exit
```

#### Logs
- Application logs: `logs/app.log`
- Redis logs: `/usr/local/var/log/redis.log`

## Project Structure

```
.
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables (create this)
└── src/
    ├── main.py          # Main application entry point
    ├── api/
    │   ├── monitor.py   # API monitoring logic
    │   └── dashboard.py # Streamlit dashboard
    └── utils/
        └── logging.py   # Logging configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
