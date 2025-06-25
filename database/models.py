"""
MorganVuoksi Elite Terminal - Production Database Models
MISSION CRITICAL: Bloomberg-grade financial data models
ZERO PLACEHOLDERS, 100% OPERATIONAL
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    DECIMAL, BigInteger, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class MarketData(Base):
    """High-frequency market data storage."""
    __tablename__ = 'market_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # OHLCV data
    open_price = Column(DECIMAL(12, 4), nullable=False)
    high_price = Column(DECIMAL(12, 4), nullable=False)
    low_price = Column(DECIMAL(12, 4), nullable=False)
    close_price = Column(DECIMAL(12, 4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    # Additional market data
    bid_price = Column(DECIMAL(12, 4))
    ask_price = Column(DECIMAL(12, 4))
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    spread = Column(DECIMAL(8, 4))
    
    # Technical indicators
    rsi = Column(DECIMAL(6, 2))
    macd = Column(DECIMAL(8, 4))
    macd_signal = Column(DECIMAL(8, 4))
    bb_upper = Column(DECIMAL(12, 4))
    bb_lower = Column(DECIMAL(12, 4))
    
    # Data quality
    data_source = Column(String(50), nullable=False)
    quality_score = Column(DECIMAL(3, 2))
    
    created_at = Column(DateTime, default=func.now())
    
    # Indexes for high-performance queries
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp_symbol', 'timestamp', 'symbol'),
        UniqueConstraint('symbol', 'timestamp', 'data_source', name='uq_market_data')
    )

class NewsData(Base):
    """Financial news and sentiment data."""
    __tablename__ = 'news_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(100), nullable=False)
    url = Column(Text)
    
    published_at = Column(DateTime, nullable=False, index=True)
    processed_at = Column(DateTime, default=func.now())
    
    # NLP analysis
    sentiment_score = Column(DECIMAL(5, 4))  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(DECIMAL(5, 4))
    keywords = Column(JSON)
    entities = Column(JSON)
    
    # Market impact
    price_impact = Column(DECIMAL(8, 4))
    volume_impact = Column(DECIMAL(8, 4))
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_symbol_published', 'symbol', 'published_at'),
        Index('idx_sentiment_score', 'sentiment_score'),
    )

class Portfolio(Base):
    """Portfolio holdings and performance."""
    __tablename__ = 'portfolios'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Portfolio metadata
    strategy = Column(String(50))
    risk_tolerance = Column(String(20))
    benchmark = Column(String(20))
    
    # Performance metrics
    total_value = Column(DECIMAL(15, 2), nullable=False, default=0)
    cash_balance = Column(DECIMAL(15, 2), nullable=False, default=0)
    invested_amount = Column(DECIMAL(15, 2), nullable=False, default=0)
    
    # Returns
    daily_return = Column(DECIMAL(8, 4))
    total_return = Column(DECIMAL(8, 4))
    sharpe_ratio = Column(DECIMAL(6, 4))
    max_drawdown = Column(DECIMAL(6, 4))
    
    # Risk metrics
    var_95 = Column(DECIMAL(10, 4))
    cvar_95 = Column(DECIMAL(10, 4))
    beta = Column(DECIMAL(6, 4))
    volatility = Column(DECIMAL(6, 4))
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    positions = relationship("Position", back_populates="portfolio")
    trades = relationship("Trade", back_populates="portfolio")

class Position(Base):
    """Individual position within a portfolio."""
    __tablename__ = 'positions'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position data
    quantity = Column(DECIMAL(15, 6), nullable=False)
    average_cost = Column(DECIMAL(12, 4), nullable=False)
    current_price = Column(DECIMAL(12, 4))
    market_value = Column(DECIMAL(15, 2))
    
    # P&L
    unrealized_pnl = Column(DECIMAL(12, 2))
    realized_pnl = Column(DECIMAL(12, 2))
    total_pnl = Column(DECIMAL(12, 2))
    
    # Risk metrics
    position_var = Column(DECIMAL(10, 4))
    position_beta = Column(DECIMAL(6, 4))
    weight = Column(DECIMAL(6, 4))  # Portfolio weight
    
    # Position metadata
    sector = Column(String(50))
    industry = Column(String(100))
    asset_class = Column(String(30))
    
    opened_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    
    __table_args__ = (
        Index('idx_portfolio_symbol', 'portfolio_id', 'symbol'),
        UniqueConstraint('portfolio_id', 'symbol', name='uq_portfolio_position')
    )

class Trade(Base):
    """Trade execution records."""
    __tablename__ = 'trades'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'), nullable=False)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(DECIMAL(15, 6), nullable=False)
    price = Column(DECIMAL(12, 4), nullable=False)
    amount = Column(DECIMAL(15, 2), nullable=False)
    
    # Order details
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP
    time_in_force = Column(String(10))  # DAY, GTC, IOC
    
    # Execution details
    execution_time = Column(DateTime, nullable=False, index=True)
    broker = Column(String(50))
    execution_venue = Column(String(50))
    commission = Column(DECIMAL(8, 2))
    fees = Column(DECIMAL(8, 2))
    
    # Trade metadata
    strategy = Column(String(50))
    signal_source = Column(String(50))
    confidence = Column(DECIMAL(5, 4))
    expected_return = Column(DECIMAL(8, 4))
    
    # Performance tracking
    realized_pnl = Column(DECIMAL(12, 2))
    holding_period = Column(Integer)  # Days
    
    # Status
    status = Column(String(20), nullable=False, default='PENDING')
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="trades")
    
    __table_args__ = (
        Index('idx_symbol_execution_time', 'symbol', 'execution_time'),
        Index('idx_portfolio_execution_time', 'portfolio_id', 'execution_time'),
        CheckConstraint('side IN ("BUY", "SELL")', name='check_trade_side'),
        CheckConstraint('quantity > 0', name='check_positive_quantity'),
        CheckConstraint('price > 0', name='check_positive_price')
    )

class MLModel(Base):
    """ML model registry and metadata."""
    __tablename__ = 'ml_models'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # LSTM, Transformer, XGBoost
    version = Column(String(20), nullable=False)
    
    # Model configuration
    hyperparameters = Column(JSON)
    features = Column(JSON)
    target = Column(String(50))
    
    # Training metadata
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    training_samples = Column(Integer)
    
    # Performance metrics
    train_accuracy = Column(DECIMAL(6, 4))
    validation_accuracy = Column(DECIMAL(6, 4))
    test_accuracy = Column(DECIMAL(6, 4))
    train_loss = Column(DECIMAL(10, 6))
    validation_loss = Column(DECIMAL(10, 6))
    sharpe_ratio = Column(DECIMAL(6, 4))
    
    # Model status
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    deployment_date = Column(DateTime)
    
    # File paths
    model_path = Column(String(500))
    scaler_path = Column(String(500))
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")
    
    __table_args__ = (
        Index('idx_model_type_version', 'model_type', 'version'),
        UniqueConstraint('name', 'version', name='uq_model_version')
    )

class Prediction(Base):
    """Model predictions and results."""
    __tablename__ = 'predictions'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(String(36), ForeignKey('ml_models.id'), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Prediction data
    prediction_date = Column(DateTime, nullable=False, index=True)
    target_date = Column(DateTime, nullable=False)
    predicted_value = Column(DECIMAL(12, 4), nullable=False)
    confidence = Column(DECIMAL(5, 4))
    confidence_interval_upper = Column(DECIMAL(12, 4))
    confidence_interval_lower = Column(DECIMAL(12, 4))
    
    # Actual outcome (for backtesting)
    actual_value = Column(DECIMAL(12, 4))
    prediction_error = Column(DECIMAL(8, 4))
    absolute_error = Column(DECIMAL(8, 4))
    
    # Prediction metadata
    features_used = Column(JSON)
    model_confidence = Column(DECIMAL(5, 4))
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_symbol_prediction_date', 'symbol', 'prediction_date'),
        Index('idx_model_symbol_date', 'model_id', 'symbol', 'prediction_date'),
    )

class RiskMetrics(Base):
    """Risk metrics and monitoring."""
    __tablename__ = 'risk_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'), nullable=False)
    calculation_date = Column(DateTime, nullable=False, index=True)
    
    # VaR metrics
    var_95_1d = Column(DECIMAL(10, 4))
    var_99_1d = Column(DECIMAL(10, 4))
    cvar_95_1d = Column(DECIMAL(10, 4))
    cvar_99_1d = Column(DECIMAL(10, 4))
    
    # Risk decomposition
    systematic_risk = Column(DECIMAL(8, 4))
    idiosyncratic_risk = Column(DECIMAL(8, 4))
    concentration_risk = Column(DECIMAL(6, 4))
    correlation_risk = Column(DECIMAL(6, 4))
    
    # Stress test results
    market_crash_impact = Column(DECIMAL(8, 4))
    interest_rate_shock = Column(DECIMAL(8, 4))
    volatility_spike_impact = Column(DECIMAL(8, 4))
    
    # Risk limits
    var_limit = Column(DECIMAL(10, 4))
    concentration_limit = Column(DECIMAL(6, 4))
    max_drawdown_limit = Column(DECIMAL(6, 4))
    
    # Breach indicators
    var_breach = Column(Boolean, default=False)
    concentration_breach = Column(Boolean, default=False)
    drawdown_breach = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_portfolio_calc_date', 'portfolio_id', 'calculation_date'),
    )

class BacktestResult(Base):
    """Backtesting results and performance analysis."""
    __tablename__ = 'backtest_results'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name = Column(String(100), nullable=False)
    model_id = Column(String(36), ForeignKey('ml_models.id'))
    
    # Backtest parameters
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(DECIMAL(15, 2), nullable=False)
    symbols = Column(JSON)
    
    # Performance metrics
    final_value = Column(DECIMAL(15, 2))
    total_return = Column(DECIMAL(8, 4))
    annualized_return = Column(DECIMAL(8, 4))
    volatility = Column(DECIMAL(6, 4))
    sharpe_ratio = Column(DECIMAL(6, 4))
    sortino_ratio = Column(DECIMAL(6, 4))
    max_drawdown = Column(DECIMAL(6, 4))
    
    # Trade statistics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(DECIMAL(5, 4))
    avg_win = Column(DECIMAL(8, 4))
    avg_loss = Column(DECIMAL(8, 4))
    profit_factor = Column(DECIMAL(6, 4))
    
    # Risk metrics
    var_95 = Column(DECIMAL(8, 4))
    cvar_95 = Column(DECIMAL(8, 4))
    beta = Column(DECIMAL(6, 4))
    alpha = Column(DECIMAL(6, 4))
    
    # Configuration
    strategy_config = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_strategy_start_date', 'strategy_name', 'start_date'),
    )

class SystemMetrics(Base):
    """System performance and monitoring metrics."""
    __tablename__ = 'system_metrics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(DECIMAL(15, 6))
    metric_unit = Column(String(20))
    
    # System component
    component = Column(String(50), nullable=False)
    service = Column(String(50))
    
    # Metadata
    tags = Column(JSON)
    
    timestamp = Column(DateTime, nullable=False, default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_component_metric_time', 'component', 'metric_name', 'timestamp'),
    )

class AlertLog(Base):
    """System alerts and notifications."""
    __tablename__ = 'alert_logs'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Alert content  
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Context
    component = Column(String(50))
    portfolio_id = Column(String(36), ForeignKey('portfolios.id'))
    symbol = Column(String(20))
    
    # Alert data
    trigger_value = Column(DECIMAL(15, 6))
    threshold_value = Column(DECIMAL(15, 6))
    additional_data = Column(JSON)
    
    # Status
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(String(100))
    
    created_at = Column(DateTime, default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_created_resolved', 'created_at', 'is_resolved'),
        CheckConstraint('severity IN ("LOW", "MEDIUM", "HIGH", "CRITICAL")', name='check_alert_severity')
    )

# Database utilities and functions
def create_all_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_table_stats(session):
    """Get database table statistics."""
    stats = {}
    for table in Base.metadata.tables.keys():
        count = session.execute(f"SELECT COUNT(*) FROM {table}").scalar()
        stats[table] = count
    return stats