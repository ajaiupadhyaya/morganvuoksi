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

- Python 3.8+
- Redis server
- API keys for monitored services (if required)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd api-monitoring-dashboard
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

4. Configure environment variables:
Create a `.env` file with your API keys:
```
WEATHER_API_KEY=your_weather_api_key
NEWS_API_KEY=your_news_api_key
```

5. Configure monitoring:
Edit `config.yaml` to add your API endpoints and monitoring settings.

## Usage

1. Start Redis server:
```bash
redis-server
```

2. Run the dashboard:
```bash
streamlit run src/main.py
```

The dashboard will be available at `http://localhost:8501`

## Configuration

The `config.yaml` file contains all configuration settings:

- Redis connection details
- Monitoring intervals
- API endpoints to monitor
- Headers and authentication

## Project Structure

```
.
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
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
