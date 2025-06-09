# API Credentials Guide

This guide lists all required and optional API credentials for the ML Trading System.

## Required APIs

### 1. Alpaca Trading API
- **Purpose**: Live market data and trading execution
- **Key Location**: `.env` file
- **Variables**: 
  - `ALPACA_API_KEY`
  - `ALPACA_API_SECRET`
- **Registration**: [Alpaca Dashboard](https://app.alpaca.markets/signup)
- **Documentation**: [Alpaca API Docs](https://alpaca.markets/docs/api-documentation/)

### 2. Polygon.io API
- **Purpose**: Historical market data, fundamentals, and news
- **Key Location**: `.env` file
- **Variables**: 
  - `POLYGON_API_KEY`
- **Registration**: [Polygon.io](https://polygon.io/dashboard/signup)
- **Documentation**: [Polygon API Docs](https://polygon.io/docs)

### 3. Yahoo Finance API
- **Purpose**: Backup market data and fundamentals
- **Key Location**: `.env` file
- **Variables**: 
  - `YAHOO_FINANCE_API_KEY` (optional)
- **Registration**: [Yahoo Finance API](https://www.yahoo.com/developer)
- **Documentation**: [Yahoo Finance API Docs](https://developer.yahoo.com/finance/)

## Optional APIs

### 1. OpenAI API
- **Purpose**: NLP analysis, report generation, and insights
- **Key Location**: `.env` file
- **Variables**: 
  - `OPENAI_API_KEY`
- **Registration**: [OpenAI Platform](https://platform.openai.com/signup)
- **Documentation**: [OpenAI API Docs](https://platform.openai.com/docs)

### 2. Slack API
- **Purpose**: Alerts and notifications
- **Key Location**: `.env` file
- **Variables**: 
  - `SLACK_BOT_TOKEN`
  - `SLACK_CHANNEL_ID`
- **Registration**: [Slack API](https://api.slack.com/apps)
- **Documentation**: [Slack API Docs](https://api.slack.com/docs)

### 3. Finnhub API
- **Purpose**: Alternative market data and news
- **Key Location**: `.env` file
- **Variables**: 
  - `FINNHUB_API_KEY`
- **Registration**: [Finnhub](https://finnhub.io/register)
- **Documentation**: [Finnhub API Docs](https://finnhub.io/docs/api)

### 4. Alpha Vantage API
- **Purpose**: Technical indicators and market data
- **Key Location**: `.env` file
- **Variables**: 
  - `ALPHA_VANTAGE_API_KEY`
- **Registration**: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Documentation**: [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)

## Cloud Storage APIs

### 1. AWS S3 (Optional)
- **Purpose**: Model storage and data backup
- **Key Location**: `.env` file
- **Variables**: 
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`
- **Registration**: [AWS Console](https://aws.amazon.com/console/)
- **Documentation**: [AWS S3 Docs](https://docs.aws.amazon.com/s3/)

### 2. Google Cloud Storage (Optional)
- **Purpose**: Alternative cloud storage
- **Key Location**: `.env` file
- **Variables**: 
  - `GOOGLE_CLOUD_PROJECT`
  - `GOOGLE_APPLICATION_CREDENTIALS`
- **Registration**: [Google Cloud Console](https://console.cloud.google.com/)
- **Documentation**: [GCS Docs](https://cloud.google.com/storage/docs)

## Security Best Practices

1. **Key Storage**
   - Never commit API keys to version control
   - Use environment variables or secure vaults
   - Rotate keys regularly
   - Use different keys for development/production

2. **Access Control**
   - Implement rate limiting
   - Use API key restrictions
   - Monitor API usage
   - Set up alerts for unusual activity

3. **Backup Strategy**
   - Maintain backup API keys
   - Document key rotation procedures
   - Store keys in multiple secure locations
   - Test backup keys regularly

## Setup Instructions

1. **Create .env File**
   ```bash
   cp .env.example .env
   ```

2. **Add API Keys**
   ```bash
   # Required APIs
   ALPACA_API_KEY=your_key_here
   ALPACA_API_SECRET=your_secret_here
   POLYGON_API_KEY=your_key_here
   
   # Optional APIs
   OPENAI_API_KEY=your_key_here
   SLACK_BOT_TOKEN=your_token_here
   SLACK_CHANNEL_ID=your_channel_here
   ```

3. **Verify Configuration**
   ```bash
   python verify_credentials.py
   ```

## Troubleshooting

1. **API Key Issues**
   - Check key format
   - Verify permissions
   - Check rate limits
   - Validate IP restrictions

2. **Connection Problems**
   - Check network
   - Verify endpoints
   - Test API directly
   - Check firewall rules

3. **Rate Limiting**
   - Monitor usage
   - Implement backoff
   - Use caching
   - Distribute requests

## Support

1. **API Support**
   - Alpaca: support@alpaca.markets
   - Polygon: support@polygon.io
   - OpenAI: support@openai.com
   - Slack: api-support@slack.com

2. **Documentation**
   - API docs
   - Rate limits
   - Best practices
   - Example code

3. **Community**
   - Stack Overflow
   - GitHub issues
   - API forums
   - Developer communities 