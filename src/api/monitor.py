"""
API monitoring system with status tracking.
"""
import time
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class APIMonitor:
    """Monitor API health and performance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            password=config.get('redis_password', None)
        )
        self.apis = config.get('apis', {})
        self.check_interval = config.get('check_interval', 60)  # seconds
        self.retention_period = config.get('retention_period', 7)  # days
    
    def check_api_health(self, api_name: str, endpoint: str,
                        headers: Optional[Dict] = None) -> Dict:
        """Check health of a specific API endpoint."""
        try:
            start_time = time.time()
            response = requests.get(endpoint, headers=headers, timeout=5)
            latency = time.time() - start_time
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'api_name': api_name,
                'endpoint': endpoint,
                'status_code': response.status_code,
                'latency': latency,
                'is_healthy': response.status_code == 200
            }
            
            # Store in Redis
            self._store_api_status(status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking API {api_name}: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'api_name': api_name,
                'endpoint': endpoint,
                'status_code': None,
                'latency': None,
                'is_healthy': False,
                'error': str(e)
            }
    
    def _store_api_status(self, status: Dict):
        """Store API status in Redis."""
        key = f"api_status:{status['api_name']}"
        self.redis_client.lpush(key, str(status))
        self.redis_client.ltrim(key, 0, 1000)  # Keep last 1000 statuses
    
    def get_api_status(self, api_name: str,
                      lookback_hours: int = 24) -> pd.DataFrame:
        """Get API status history."""
        key = f"api_status:{api_name}"
        statuses = self.redis_client.lrange(key, 0, -1)
        
        if not statuses:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([eval(s) for s in statuses])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by lookback period
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        df = df[df['timestamp'] >= cutoff]
        
        return df.sort_values('timestamp')
    
    def get_api_metrics(self, api_name: str,
                       lookback_hours: int = 24) -> Dict:
        """Calculate API performance metrics."""
        df = self.get_api_status(api_name, lookback_hours)
        
        if df.empty:
            return {}
        
        # Calculate metrics
        uptime = df['is_healthy'].mean()
        avg_latency = df['latency'].mean()
        p95_latency = df['latency'].quantile(0.95)
        error_rate = 1 - uptime
        
        # Calculate hourly metrics
        hourly_metrics = df.set_index('timestamp').resample('H').agg({
            'is_healthy': 'mean',
            'latency': ['mean', 'std'],
            'status_code': 'count'
        })
        
        return {
            'uptime': uptime,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'error_rate': error_rate,
            'hourly_metrics': hourly_metrics
        }
    
    def monitor_all_apis(self):
        """Monitor all configured APIs."""
        for api_name, api_config in self.apis.items():
            status = self.check_api_health(
                api_name,
                api_config['endpoint'],
                api_config.get('headers')
            )
            
            if not status['is_healthy']:
                logger.warning(f"API {api_name} is unhealthy: {status}")
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive API status report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'apis': {}
        }
        
        for api_name in self.apis.keys():
            metrics = self.get_api_metrics(api_name)
            report['apis'][api_name] = metrics
        
        return report
    
    def start_monitoring(self):
        """Start continuous API monitoring."""
        while True:
            try:
                self.monitor_all_apis()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in API monitoring: {str(e)}")
                time.sleep(self.check_interval)
    
    def cleanup_old_data(self):
        """Clean up old API status data."""
        cutoff = datetime.now() - timedelta(days=self.retention_period)
        cutoff_str = cutoff.isoformat()
        
        for api_name in self.apis.keys():
            key = f"api_status:{api_name}"
            statuses = self.redis_client.lrange(key, 0, -1)
            
            # Filter out old statuses
            new_statuses = [
                s for s in statuses
                if eval(s)['timestamp'] >= cutoff_str
            ]
            
            # Update Redis
            if new_statuses:
                self.redis_client.delete(key)
                self.redis_client.lpush(key, *new_statuses)
            else:
                self.redis_client.delete(key)
    
    def get_api_dashboard_data(self) -> Dict:
        """Get data for API monitoring dashboard."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'apis': {}
        }
        
        for api_name in self.apis.keys():
            # Get recent status
            df = self.get_api_status(api_name, lookback_hours=24)
            
            if not df.empty:
                # Calculate metrics
                metrics = self.get_api_metrics(api_name)
                
                # Get status history
                status_history = df[['timestamp', 'is_healthy', 'latency']].to_dict('records')
                
                dashboard_data['apis'][api_name] = {
                    'metrics': metrics,
                    'status_history': status_history,
                    'current_status': status_history[0] if status_history else None
                }
        
        return dashboard_data 