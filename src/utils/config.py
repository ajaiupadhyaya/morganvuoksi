"""
Configuration management for the quantitative finance system.
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager with environment-based overrides."""
    
    def __init__(self):
        """Initialize configuration with defaults and environment overrides."""
        self.config_dir = Path(__file__).parent.parent.parent / 'config'
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML files."""
        config = {}
        
        # Load base config
        base_config_path = self.config_dir / 'base.yaml'
        if base_config_path.exists():
            with open(base_config_path) as f:
                config.update(yaml.safe_load(f))
        
        # Load environment-specific config
        env = os.getenv('ENV', 'development')
        env_config_path = self.config_dir / f'{env}.yaml'
        if env_config_path.exists():
            with open(env_config_path) as f:
                config.update(yaml.safe_load(f))
        
        return config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # API Keys
        self.config['ALPHA_VANTAGE_API_KEY'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.config['POLYGON_API_KEY'] = os.getenv('POLYGON_API_KEY')
        self.config['IEX_API_KEY'] = os.getenv('IEX_API_KEY')
        self.config['FRED_API_KEY'] = os.getenv('FRED_API_KEY')
        self.config['NEWS_API_KEY'] = os.getenv('NEWS_API_KEY')
        self.config['REDDIT_CLIENT_ID'] = os.getenv('REDDIT_CLIENT_ID')
        self.config['REDDIT_CLIENT_SECRET'] = os.getenv('REDDIT_CLIENT_SECRET')
        self.config['TWITTER_API_KEY'] = os.getenv('TWITTER_API_KEY')
        self.config['TWITTER_API_SECRET'] = os.getenv('TWITTER_API_SECRET')
        
        # Database
        self.config['REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.config['INFLUXDB_URL'] = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.config['INFLUXDB_TOKEN'] = os.getenv('INFLUXDB_TOKEN')
        
        # Model Settings
        self.config['MODEL_CACHE_DIR'] = os.getenv('MODEL_CACHE_DIR', 'models/cache')
        self.config['USE_GPU'] = os.getenv('USE_GPU', 'false').lower() == 'true'
        
        # Logging
        self.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'INFO')
        self.config['LOG_FORMAT'] = os.getenv('LOG_FORMAT', 'json')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        return self.config[key]

# Global configuration instance
config = Config()

def get_config() -> Dict[str, Any]:
    """Get the global configuration instance."""
    return config.config 
