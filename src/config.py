# src/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
import logging
from dotenv import load_dotenv
import json

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

if __name__ == "__main__":
    main()