# src/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import logging
from dotenv import load_dotenv
import json
import yaml

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration."""
    alpaca_key: str = os.getenv('ALPACA_API_KEY', '')
    alpaca_secret: str = os.getenv('ALPACA_API_SECRET', '')
    polygon_key: str = os.getenv('POLYGON_API_KEY', '')
    newsapi_key: str = os.getenv('NEWSAPI_KEY', '')
    finnhub_key: str = os.getenv('FINNHUB_KEY', '')
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_KEY', '')

@dataclass
class Paths:
    """Path configuration."""
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / 'data'
    raw_data_dir: Path = data_dir / 'raw'
    processed_data_dir: Path = data_dir / 'processed'
    models_dir: Path = base_dir / 'models'
    logs_dir: Path = base_dir / 'logs'
    config_dir: Path = base_dir / 'config'
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.logs_dir,
            self.config_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class MarketDataConfig:
    """Market data configuration."""
    data_providers: List[str] = ['alpaca', 'polygon', 'yfinance']
    update_frequency: str = '1min'
    max_retries: int = 3
    timeout: int = 30
    cache_size: int = 1000
    min_liquidity: float = 1e6
    min_market_cap: float = 1e9
    max_spread: float = 0.01
    min_volume: float = 1e5
    default_interval: str = '1d'
    default_provider: str = 'alpaca'
    symbols: List[str] = None
    
    def __post_init__(self):
        """Load symbols from config file if not provided."""
        if self.symbols is None:
            try:
                with open(Paths().config_dir / 'symbols.json', 'r') as f:
                    self.symbols = json.load(f)
            except FileNotFoundError:
                logger.warning("symbols.json not found, using default symbols")
                self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']

@dataclass
class TradingConfig:
    """Trading configuration."""
    default_commission: float = 0.001
    max_position_size: float = 0.2
    risk_free_rate: float = 0.02
    leverage: float = 1.0
    min_holding_period: int = 1
    max_holding_period: int = 252
    rebalance_frequency: str = '1d'
    max_drawdown: float = 0.1
    volatility_threshold: float = 0.3
    correlation_threshold: float = 0.7
    circuit_breaker_threshold: float = 0.05
    min_probability: float = 0.6
    max_turnover: float = 0.1
    max_leverage: float = 2.0
    max_sector_exposure: float = 0.3
    max_factor_exposure: float = 0.2
    min_liquidity_score: float = 0.5
    max_volatility: float = 0.4
    min_diversification: float = 0.5

@dataclass
class ModelConfig:
    """Model configuration."""
    n_models: int = 3
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 10
    n_estimators: int = 100
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0
    reg_alpha: float = 0
    reg_lambda: float = 1
    scale_pos_weight: float = 1
    objective: str = 'binary:logistic'
    eval_metric: str = 'auc'
    early_stopping_rounds: int = 10
    verbose: bool = False

@dataclass
class Config:
    """Main configuration."""
    api: APIConfig = APIConfig()
    paths: Paths = Paths()
    market_data: MarketDataConfig = MarketDataConfig()
    trading: TradingConfig = TradingConfig()
    model: ModelConfig = ModelConfig()
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        if path is None:
            path = self.paths.config_dir / 'config.json'
        
        config_dict = {
            'api': self.api.__dict__,
            'market_data': self.market_data.__dict__,
            'trading': self.trading.__dict__,
            'model': self.model.__dict__
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            path: Path to load configuration from
        
        Returns:
            Config object
        """
        if path is None:
            path = Paths().config_dir / 'config.json'
        
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            return cls(
                api=APIConfig(**config_dict['api']),
                market_data=MarketDataConfig(**config_dict['market_data']),
                trading=TradingConfig(**config_dict['trading']),
                model=ModelConfig(**config_dict['model'])
            )
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found at {path}, using defaults")
            return cls()

# Initialize configuration
config = Config()

def main():
    """Main execution function."""
    try:
        # Save default configuration
        config.save()
        
        logger.info("Configuration initialized successfully")
        
    except Exception as e:
        logger.error(f"Configuration initialization failed: {str(e)}")
        raise

def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()

def get_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables and config files.
    
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_env()
    
    # Base configuration
    config = {
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD')
        },
        'kafka': {
            'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            'topic': os.getenv('KAFKA_TOPIC', 'market_data')
        },
        'influxdb': {
            'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
            'token': os.getenv('INFLUXDB_TOKEN'),
            'org': os.getenv('INFLUXDB_ORG'),
            'bucket': os.getenv('INFLUXDB_BUCKET', 'market_data')
        },
        'ray': {
            'address': os.getenv('RAY_ADDRESS', 'auto'),
            'namespace': os.getenv('RAY_NAMESPACE', 'trading')
        },
        'zmq': {
            'port': int(os.getenv('ZMQ_PORT', 5555))
        },
        'ib': {
            'host': os.getenv('IB_HOST', '127.0.0.1'),
            'port': int(os.getenv('IB_PORT', 7497)),
            'client_id': int(os.getenv('IB_CLIENT_ID', 1))
        },
        'alpaca': {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'api_secret': os.getenv('ALPACA_API_SECRET'),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        },
        'models': {
            'lstm': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001
            },
            'transformer': {
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 2048,
                'dropout': 0.1
            },
            'tft': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001
            },
            'nbeats': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001
            },
            'ppo': {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'eps_clip': 0.2,
                'K_epochs': 10
            },
            'maml': {
                'learning_rate': 0.001,
                'meta_learning_rate': 0.01,
                'num_tasks': 5,
                'num_steps': 5
            }
        },
        'research': {
            'fama_french': True,
            'statistical': True,
            'risk_analytics': True,
            'regime_switching': True,
            'cointegration': True
        },
        'trading': {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', 100000)),
            'max_leverage': float(os.getenv('MAX_LEVERAGE', 2.0)),
            'stop_loss': float(os.getenv('STOP_LOSS', 0.02)),
            'take_profit': float(os.getenv('TAKE_PROFIT', 0.04))
        },
        'logging': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    # Load additional configuration from YAML files
    config_files = [
        'config/database.yaml',
        'config/models.yaml',
        'config/trading.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
    
    return config

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Current configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(config, updates)

def save_config(config: Dict[str, Any], filename: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        filename: Output filename
    """
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Print configuration
    print(yaml.dump(config, default_flow_style=False))