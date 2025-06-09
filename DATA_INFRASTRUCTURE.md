# Data Infrastructure Guide

This guide outlines the institution-grade data infrastructure for the ML Trading System.

## Data Sources

### Tier 1 (Essential)

#### 1. Alpha Vantage
```python
# data/sources/alpha_vantage.py
class AlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter(max_calls=5, time_window=60)
    
    async def get_ohlcv(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get OHLCV data with rate limiting."""
        async with self.rate_limiter:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    return self._process_ohlcv(data)
    
    async def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data with rate limiting."""
        async with self.rate_limiter:
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    return await response.json()
```

#### 2. Yahoo Finance
```python
# data/sources/yahoo_finance.py
class YahooFinanceClient:
    def __init__(self):
        self.cache = RedisCache()
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data with caching."""
        cache_key = f"yahoo:{symbol}:{start_date}:{end_date}"
        
        # Check cache
        if cached_data := await self.cache.get(cache_key):
            return pd.DataFrame(cached_data)
        
        # Fetch data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Cache result
        await self.cache.set(cache_key, data.to_dict(), expire=3600)
        
        return data
```

#### 3. FRED API
```python
# data/sources/fred.py
class FREDClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
    
    async def get_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """Get economic data series."""
        return self.fred.get_series(series_id, start_date, end_date)
    
    async def get_yield_curve(self) -> pd.DataFrame:
        """Get yield curve data."""
        tenors = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        curves = {}
        for tenor in tenors:
            curves[tenor] = await self.get_series(tenor, start_date='2020-01-01', end_date='today')
        return pd.DataFrame(curves)
```

### Tier 2 (Professional)

#### 1. IEX Cloud
```python
# data/sources/iex.py
class IEXClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://cloud.iexapis.com/v1"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
    
    async def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote with rate limiting."""
        async with self.rate_limiter:
            url = f"{self.base_url}/stock/{symbol}/quote"
            params = {"token": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    return await response.json()
    
    async def get_options_chain(self, symbol: str) -> pd.DataFrame:
        """Get options chain with rate limiting."""
        async with self.rate_limiter:
            url = f"{self.base_url}/stock/{symbol}/options"
            params = {"token": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return pd.DataFrame(data)
```

#### 2. Polygon.io
```python
# data/sources/polygon.py
class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        self.rate_limiter = RateLimiter(max_calls=5, time_window=60)
    
    async def get_trades(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        """Get trade data with rate limiting."""
        async with self.rate_limiter:
            url = f"{self.base_url}/trades/{symbol}/{start_time}/{end_time}"
            params = {"apiKey": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return pd.DataFrame(data['results'])
```

#### 3. Quandl/Nasdaq Data Link
```python
# data/sources/quandl.py
class QuandlClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        quandl.ApiConfig.api_key = api_key
    
    async def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Get fundamental data."""
        return quandl.get_table('SHARADAR/SF1', 
                              ticker=symbol,
                              dimension='ARQ')
    
    async def get_alternative_data(self, dataset: str, **kwargs) -> pd.DataFrame:
        """Get alternative data."""
        return quandl.get(dataset, **kwargs)
```

### Tier 3 (Alternative Data)

#### 1. News API
```python
# data/sources/news.py
class NewsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
    
    async def get_news(self, query: str, from_date: str, to_date: str) -> List[Dict]:
        """Get news articles with rate limiting."""
        async with self.rate_limiter:
            url = f"{self.base_url}/everything"
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "apiKey": self.api_key,
                "language": "en",
                "sortBy": "relevancy"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data['articles']
```

#### 2. Reddit API
```python
# data/sources/reddit.py
class RedditClient:
    def __init__(self, client_id: str, client_secret: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="ML Trading System"
        )
    
    async def get_sentiment(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        """Get sentiment data from subreddit."""
        posts = []
        for submission in self.reddit.subreddit(subreddit).hot(limit=limit):
            posts.append({
                'title': submission.title,
                'score': submission.score,
                'created_utc': submission.created_utc,
                'num_comments': submission.num_comments
            })
        return pd.DataFrame(posts)
```

#### 3. Twitter API
```python
# data/sources/twitter.py
class TwitterClient:
    def __init__(self, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token)
    
    async def get_sentiment(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """Get sentiment data from Twitter."""
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['created_at', 'public_metrics']
        )
        return pd.DataFrame(tweets.data)
```

## Data Pipeline

### 1. Data Ingestion
```python
# data/pipeline/ingestion.py
class DataIngestionPipeline:
    def __init__(self):
        self.sources = {
            'alpha_vantage': AlphaVantageClient(API_KEY),
            'yahoo': YahooFinanceClient(),
            'fred': FREDClient(API_KEY),
            'iex': IEXClient(API_KEY),
            'polygon': PolygonClient(API_KEY),
            'quandl': QuandlClient(API_KEY),
            'news': NewsClient(API_KEY),
            'reddit': RedditClient(CLIENT_ID, CLIENT_SECRET),
            'twitter': TwitterClient(BEARER_TOKEN)
        }
    
    async def ingest_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """Ingest data from specified source."""
        source = self.sources[data_type]
        data = await source.get_data(**kwargs)
        return self._validate_data(data)
```

### 2. Data Validation
```python
# data/pipeline/validation.py
class DataValidator:
    def __init__(self):
        self.validators = {
            'ohlcv': OHLCVValidator(),
            'fundamentals': FundamentalsValidator(),
            'alternative': AlternativeDataValidator()
        }
    
    def validate(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate data quality."""
        validator = self.validators[data_type]
        return validator.validate(data)
```

### 3. Data Storage
```python
# data/pipeline/storage.py
class DataStorage:
    def __init__(self):
        self.timescale = TimescaleDB()
        self.redis = RedisCache()
    
    async def store_data(self, data: pd.DataFrame, data_type: str):
        """Store data in appropriate database."""
        if data_type in ['ohlcv', 'trades']:
            await self.timescale.store(data)
        else:
            await self.redis.store(data)
```

## Data Quality

### 1. Gap Detection
```python
# data/quality/gap_detection.py
class GapDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def detect_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """Detect data gaps."""
        gaps = []
        for col in data.columns:
            if data[col].isnull().sum() / len(data) > self.threshold:
                gaps.append({
                    'column': col,
                    'gap_percentage': data[col].isnull().sum() / len(data)
                })
        return gaps
```

### 2. Outlier Detection
```python
# data/quality/outlier_detection.py
class OutlierDetector:
    def __init__(self, method: str = 'zscore'):
        self.method = method
    
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in data."""
        if self.method == 'zscore':
            return self._zscore_detection(data)
        elif self.method == 'iqr':
            return self._iqr_detection(data)
```

### 3. Corporate Actions
```python
# data/quality/corporate_actions.py
class CorporateActionHandler:
    def __init__(self):
        self.actions = {
            'splits': SplitHandler(),
            'dividends': DividendHandler(),
            'mergers': MergerHandler()
        }
    
    def handle_actions(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle corporate actions in data."""
        for action_type, handler in self.actions.items():
            data = handler.adjust(data, symbol)
        return data
```

## Data Lineage

### 1. Tracking
```python
# data/lineage/tracking.py
class DataLineageTracker:
    def __init__(self):
        self.db = LineageDB()
    
    async def track_data(self, data: pd.DataFrame, source: str, timestamp: datetime):
        """Track data lineage."""
        lineage = {
            'source': source,
            'timestamp': timestamp,
            'rows': len(data),
            'columns': list(data.columns),
            'checksum': self._compute_checksum(data)
        }
        await self.db.store(lineage)
```

### 2. Versioning
```python
# data/lineage/versioning.py
class DataVersioner:
    def __init__(self):
        self.db = VersionDB()
    
    async def version_data(self, data: pd.DataFrame, version: str):
        """Version data."""
        version_info = {
            'version': version,
            'timestamp': datetime.now(),
            'schema': self._get_schema(data),
            'stats': self._compute_stats(data)
        }
        await self.db.store(version_info)
```

## Implementation Guide

### 1. Setup
```python
# config/data_config.py
def setup_data_infrastructure():
    """Configure data infrastructure."""
    # Set up databases
    setup_timescale()
    setup_redis()
    
    # Configure API clients
    setup_api_clients()
    
    # Initialize pipelines
    setup_pipelines()
```

### 2. Usage
```python
# examples/data_usage.py
async def main():
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    
    # Ingest data
    ohlcv_data = await pipeline.ingest_data('alpha_vantage', 
                                          symbol='AAPL',
                                          interval='1min')
    
    # Validate data
    validator = DataValidator()
    validated_data = validator.validate(ohlcv_data, 'ohlcv')
    
    # Store data
    storage = DataStorage()
    await storage.store_data(validated_data, 'ohlcv')
```

## Best Practices

1. **Data Quality**
   - Validate all data
   - Handle missing values
   - Detect outliers
   - Track corporate actions

2. **Performance**
   - Use async/await
   - Implement caching
   - Rate limit APIs
   - Batch processing

3. **Reliability**
   - Handle errors
   - Implement retries
   - Monitor health
   - Track lineage

4. **Security**
   - Secure API keys
   - Encrypt data
   - Audit access
   - Monitor usage

## Monitoring

1. **Data Quality**
   - Gap detection
   - Outlier detection
   - Corporate actions
   - Data validation

2. **Performance**
   - API latency
   - Cache hit rate
   - Processing time
   - Storage usage

3. **Reliability**
   - Error rates
   - Retry counts
   - Health status
   - Lineage tracking

## Future Enhancements

1. **Data Sources**
   - Add more APIs
   - Enhance validation
   - Improve caching
   - Add compression

2. **Processing**
   - Add streaming
   - Enhance batching
   - Improve validation
   - Add transformation

3. **Storage**
   - Add compression
   - Enhance indexing
   - Improve querying
   - Add backup 