"""
Logging utilities for the trading system.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create file handler
    file_handler = RotatingFileHandler(
        f'logs/{name}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_trade(logger: logging.Logger, trade: dict) -> None:
    """
    Log trade details.
    
    Args:
        logger: Logger instance
        trade: Trade details
    """
    logger.info(
        f"Trade executed - Symbol: {trade['symbol']}, "
        f"Action: {trade['action']}, "
        f"Quantity: {trade['quantity']}, "
        f"Price: {trade['price']}, "
        f"Time: {trade['time']}"
    )

def log_error(logger: logging.Logger, error: Exception, context: dict = None) -> None:
    """
    Log error with context.
    
    Args:
        logger: Logger instance
        error: Exception to log
        context: Additional context
    """
    if context:
        logger.error(
            f"Error: {str(error)}, Context: {context}"
        )
    else:
        logger.error(f"Error: {str(error)}")

def log_performance(logger: logging.Logger, metrics: dict) -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Performance metrics
    """
    logger.info(
        f"Performance metrics - "
        f"Portfolio Value: {metrics['portfolio_value']}, "
        f"Positions: {metrics['positions']}, "
        f"Timestamp: {metrics['timestamp']}"
    )

def log_system_status(logger: logging.Logger, status: dict) -> None:
    """
    Log system status.
    
    Args:
        logger: Logger instance
        status: System status
    """
    logger.info(
        f"System status - "
        f"CPU Usage: {status['cpu_usage']}%, "
        f"Memory Usage: {status['memory_usage']}%, "
        f"Disk Usage: {status['disk_usage']}%, "
        f"Network Latency: {status['network_latency']}ms"
    )

def log_api_request(logger: logging.Logger, request: dict) -> None:
    """
    Log API request.
    
    Args:
        logger: Logger instance
        request: API request details
    """
    logger.info(
        f"API request - "
        f"Method: {request['method']}, "
        f"Endpoint: {request['endpoint']}, "
        f"Status: {request['status']}, "
        f"Latency: {request['latency']}ms"
    )

def log_model_training(logger: logging.Logger, training: dict) -> None:
    """
    Log model training details.
    
    Args:
        logger: Logger instance
        training: Training details
    """
    logger.info(
        f"Model training - "
        f"Model: {training['model']}, "
        f"Epoch: {training['epoch']}, "
        f"Loss: {training['loss']}, "
        f"Accuracy: {training['accuracy']}"
    )

def log_data_processing(logger: logging.Logger, processing: dict) -> None:
    """
    Log data processing details.
    
    Args:
        logger: Logger instance
        processing: Processing details
    """
    logger.info(
        f"Data processing - "
        f"Type: {processing['type']}, "
        f"Records: {processing['records']}, "
        f"Duration: {processing['duration']}s"
    )

def log_backtest(logger: logging.Logger, backtest: dict) -> None:
    """
    Log backtest results.
    
    Args:
        logger: Logger instance
        backtest: Backtest results
    """
    logger.info(
        f"Backtest results - "
        f"Strategy: {backtest['strategy']}, "
        f"Return: {backtest['return']}%, "
        f"Sharpe: {backtest['sharpe']}, "
        f"Max Drawdown: {backtest['max_drawdown']}%"
    )

def log_risk_metrics(logger: logging.Logger, risk: dict) -> None:
    """
    Log risk metrics.
    
    Args:
        logger: Logger instance
        risk: Risk metrics
    """
    logger.info(
        f"Risk metrics - "
        f"VaR 95%: {risk['var_95']}%, "
        f"VaR 99%: {risk['var_99']}%, "
        f"Expected Shortfall: {risk['expected_shortfall']}%, "
        f"Beta: {risk['beta']}"
    )

def log_portfolio(logger: logging.Logger, portfolio: dict) -> None:
    """
    Log portfolio details.
    
    Args:
        logger: Logger instance
        portfolio: Portfolio details
    """
    logger.info(
        f"Portfolio update - "
        f"Value: {portfolio['value']}, "
        f"Positions: {portfolio['positions']}, "
        f"Cash: {portfolio['cash']}, "
        f"Leverage: {portfolio['leverage']}"
    )

if __name__ == "__main__":
    # Example usage
    logger = setup_logger('example')
    
    # Log different types of events
    log_trade(logger, {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.0,
        'time': datetime.now()
    })
    
    log_error(logger, ValueError('Invalid input'), {'context': 'data processing'})
    
    log_performance(logger, {
        'portfolio_value': 1000000,
        'positions': 10,
        'timestamp': datetime.now()
    })
    
    log_system_status(logger, {
        'cpu_usage': 50,
        'memory_usage': 60,
        'disk_usage': 70,
        'network_latency': 100
    }) 
