# Deployment Guide

This guide provides step-by-step instructions for deploying the ML Trading System in various environments.

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Heroku CLI (for cloud deployment)
- Git
- API keys for data providers (Alpaca, Polygon)

## Local Development Setup

1. **Clone the Repository**
```bash
git clone <repository-url>
cd ml-trading-system
```

2. **Set Up Python Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure Environment Variables**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required variables:
# - ALPACA_API_KEY
# - ALPACA_API_SECRET
# - POLYGON_API_KEY
# - SLACK_WEBHOOK_URL (optional)
```

4. **Run Locally**
```bash
# Start the system
./run.sh
```

## Docker Deployment

1. **Build and Run with Docker Compose**
```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

2. **Access Services**
- Dashboard: http://localhost:8050
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

3. **Stop Services**
```bash
docker-compose down
```

## Heroku Deployment

1. **Install Heroku CLI**
```bash
# On Windows (with Chocolatey):
choco install heroku-cli

# On macOS:
brew tap heroku/brew && brew install heroku

# On Ubuntu/Debian:
sudo snap install --classic heroku
```

2. **Login to Heroku**
```bash
heroku login
```

3. **Create Heroku App**
```bash
# Create new app
heroku create your-app-name

# Set buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku-community/apt
```

4. **Configure Environment Variables**
```bash
# Set required environment variables
heroku config:set ALPACA_API_KEY=your_key
heroku config:set ALPACA_API_SECRET=your_secret
heroku config:set POLYGON_API_KEY=your_key
heroku config:set SLACK_WEBHOOK_URL=your_webhook

# Set optional variables
heroku config:set DEBUG=False
heroku config:set LOG_LEVEL=INFO
```

5. **Deploy to Heroku**
```bash
# Add Heroku remote
heroku git:remote -a your-app-name

# Deploy
git push heroku main
```

6. **Monitor Deployment**
```bash
# View logs
heroku logs --tail

# Check app status
heroku ps
```

## Monitoring Setup

1. **Access Grafana**
- Open http://localhost:3000 (or your Heroku URL)
- Default credentials: admin/admin
- Change password on first login

2. **Configure Prometheus Data Source**
- Go to Configuration > Data Sources
- Add Prometheus data source
- URL: http://prometheus:9090
- Access: Server (default)

3. **Import Dashboards**
- Go to Dashboards > Import
- Upload `grafana/dashboards/trading_system.json`
- Select Prometheus data source

4. **Configure Alerts**
- Go to Alerting > Notification channels
- Add Slack channel
- Configure alert rules in dashboards

## SSL/HTTPS Setup (Heroku)

1. **Enable SSL**
```bash
# Add SSL to Heroku app
heroku certs:auto:enable
```

2. **Verify SSL**
```bash
# Check SSL status
heroku certs:auto
```

## Scaling and Maintenance

1. **Scale Heroku Dynos**
```bash
# Scale web dynos
heroku ps:scale web=2

# Scale worker dynos
heroku ps:scale worker=1
```

2. **Database Maintenance**
```bash
# Backup database
heroku pg:backups:capture

# Restore database
heroku pg:backups:restore
```

3. **Log Management**
```bash
# View recent logs
heroku logs -n 100

# View specific service logs
heroku logs --source app
```

## Troubleshooting

1. **Common Issues**
- API connection failures
- Model performance degradation
- Memory leaks
- Service crashes

2. **Debug Commands**
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart [service_name]
```

3. **Health Checks**
- Dashboard: http://localhost:8050/health
- System Status: http://localhost:8050/status

## Security Considerations

1. **API Keys**
- Never commit API keys to version control
- Rotate keys regularly
- Use environment variables

2. **Access Control**
- Secure Grafana with strong passwords
- Limit Prometheus access
- Use HTTPS in production

3. **Data Protection**
- Encrypt sensitive data
- Regular backups
- Access logging

## Backup and Recovery

1. **Regular Backups**
```bash
# Backup configuration
tar -czf config_backup.tar.gz config/

# Backup models
tar -czf models_backup.tar.gz models/
```

2. **Recovery Procedure**
```bash
# Restore configuration
tar -xzf config_backup.tar.gz

# Restore models
tar -xzf models_backup.tar.gz
```

## Support and Maintenance

1. **Regular Updates**
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update Docker images
docker-compose pull
```

2. **Monitoring**
- Check Grafana dashboards daily
- Review error logs
- Monitor system metrics

3. **Contact**
- For issues: Create GitHub issue
- For emergencies: Contact system administrator 