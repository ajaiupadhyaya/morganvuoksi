# src/execution/simulate.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, time
import logging
import seaborn as sns
from src.config import config
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    """Execution configuration."""
    slippage_model: str = 'sqrt'  # 'sqrt' or 'linear'
    market_impact_coef: float = 0.1
    min_trade_size: float = 1000
    max_trade_size: float = 100000
    max_spread: float = 0.01
    min_liquidity: float = 1e6
    max_daily_turnover: float = 0.1
    max_position_size: float = 0.1
    min_holding_period: int = 5
    max_holding_period: int = 20
    circuit_breaker_threshold: float = 0.1
    volatility_threshold: float = 0.5
    correlation_threshold: float = 0.7
    dark_pool_ratio: float = 0.2
    vwap_window: int = 30
    twap_interval: int = 5
    max_drawdown: float = 0.15
    vwap_window: int = 30
    twap_window: int = 60

class ExecutionError(Exception):
    """Custom exception for execution errors."""
    pass

class OrderManager:
    """Manages order execution and tracking."""
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        initial_cash: float = 1e6
    ):
        """
        Initialize order manager.
        
        Args:
            config: Execution configuration
            initial_cash: Initial cash balance
        """
        self.config = config or ExecutionConfig()
        self.cash = initial_cash
        self.positions = {}
        self.orders = []
        self.trades = []
        self.metrics = {}
    
    def check_risk_limits(
        self,
        symbol: str,
        size: float,
        price: float,
        volatility: float,
        correlations: Dict[str, float],
        drawdown: float
    ) -> bool:
        """
        Check if trade violates risk limits.
        
        Args:
            symbol: Asset symbol
            size: Order size
            price: Current price
            volatility: Asset volatility
            correlations: Asset correlations
            drawdown: Current drawdown
        
        Returns:
            True if trade is allowed
        """
        try:
            # Check position size
            position_value = abs(self.positions.get(symbol, 0) * price)
            if position_value + size * price > self.config.max_position_size * self.cash:
                logger.warning(f"Position size limit exceeded for {symbol}")
                return False
            
            # Check volatility
            if volatility > self.config.volatility_threshold:
                logger.warning(f"Volatility limit exceeded for {symbol}")
                return False
            
            # Check correlations
            for other_symbol, corr in correlations.items():
                if other_symbol in self.positions and abs(corr) > self.config.correlation_threshold:
                    logger.warning(f"Correlation limit exceeded for {symbol} and {other_symbol}")
                    return False
            
            # Check drawdown
            if drawdown > self.config.circuit_breaker_threshold:
                logger.warning(f"Drawdown limit exceeded")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def check_circuit_breaker(
        self,
        symbol: str,
        price: float,
        prev_price: float
    ) -> bool:
        """
        Check if price movement triggers circuit breaker.
        
        Args:
            symbol: Asset symbol
            price: Current price
            prev_price: Previous price
        
        Returns:
            True if trading is allowed
        """
        try:
            price_change = abs(price - prev_price) / prev_price
            
            if price_change > self.config.circuit_breaker_threshold:
                logger.warning(f"Circuit breaker triggered for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {str(e)}")
            return False
    
    def calculate_order_size(
        self,
        symbol: str,
        target_size: float,
        price: float,
        volatility: float,
        time_of_day: float
    ) -> float:
        """
        Calculate order size with risk adjustments.
        
        Args:
            symbol: Asset symbol
            target_size: Target position size
            price: Current price
            volatility: Asset volatility
            time_of_day: Time of day (0-1)
        
        Returns:
            Adjusted order size
        """
        try:
            # Base size
            size = target_size
            
            # Volatility adjustment
            vol_ratio = volatility / self.config.volatility_threshold
            size *= (1 - vol_ratio)
            
            # Time of day adjustment
            if time_of_day < 0.1 or time_of_day > 0.9:  # Market open/close
                size *= 0.5
            
            # Position size limits
            size = min(size, self.config.max_trade_size)
            size = max(size, self.config.min_trade_size)
            
            # Cash constraint
            max_size = self.cash * self.config.max_position_size / price
            size = min(size, max_size)
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating order size: {str(e)}")
            return 0
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'day',
        dark_pool: bool = False
    ) -> Dict:
        """
        Execute order with advanced order types.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            size: Order size
            price: Current price
            order_type: Order type ('market', 'limit', 'stop', 'twap', 'vwap')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force ('day', 'gtc', 'ioc', 'fok')
            dark_pool: Whether to use dark pool
        
        Returns:
            Order execution details
        """
        try:
            # Check risk limits
            if not self.check_risk_limits(
                symbol,
                size,
                price,
                volatility=0.1,  # Placeholder
                correlations={},  # Placeholder
                drawdown=0.0  # Placeholder
            ):
                return {
                    'status': 'rejected',
                    'reason': 'risk_limit_exceeded'
                }
            
            # Calculate execution price
            if dark_pool:
                # Dark pool execution
                execution_price = price * (1 - 0.0001)  # Small price improvement
                fill_ratio = 0.8  # Partial fill
            else:
                # Regular execution
                if order_type == 'market':
                    execution_price = price
                    fill_ratio = 1.0
                elif order_type == 'limit':
                    if (side == 'buy' and price <= limit_price) or \
                       (side == 'sell' and price >= limit_price):
                        execution_price = limit_price
                        fill_ratio = 1.0
                    else:
                        return {
                            'status': 'pending',
                            'reason': 'limit_not_met'
                        }
                elif order_type == 'stop':
                    if (side == 'buy' and price >= stop_price) or \
                       (side == 'sell' and price <= stop_price):
                        execution_price = price
                        fill_ratio = 1.0
                    else:
                        return {
                            'status': 'pending',
                            'reason': 'stop_not_triggered'
                        }
                elif order_type in ['twap', 'vwap']:
                    # Simulate TWAP/VWAP execution
                    execution_price, fill_ratio = self._simulate_twap_vwap(
                        symbol,
                        side,
                        size,
                        price,
                        order_type
                    )
                else:
                    raise ValueError(f"Unsupported order type: {order_type}")
            
            # Calculate fill size
            fill_size = size * fill_ratio
            
            # Update positions
            if side == 'buy':
                self.positions[symbol] = self.positions.get(symbol, 0) + fill_size
                self.cash -= fill_size * execution_price
            else:
                self.positions[symbol] = self.positions.get(symbol, 0) - fill_size
                self.cash += fill_size * execution_price
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'size': fill_size,
                'price': execution_price,
                'order_type': order_type,
                'dark_pool': dark_pool,
                'fill_ratio': fill_ratio
            }
            self.trades.append(trade)
            
            return {
                'status': 'filled',
                'trade': trade
            }
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return {
                'status': 'error',
                'reason': str(e)
            }
    
    def _simulate_twap_vwap(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str
    ) -> Tuple[float, float]:
        """
        Simulate TWAP/VWAP execution.
        
        Args:
            symbol: Asset symbol
            side: Order side
            size: Order size
            price: Current price
            order_type: Order type ('twap' or 'vwap')
        
        Returns:
            Tuple of (execution price, fill ratio)
        """
        try:
            if order_type == 'twap':
                # Time-weighted average price
                n_intervals = self.config.twap_interval
                interval_size = size / n_intervals
                
                prices = []
                for i in range(n_intervals):
                    # Simulate price movement
                    price_change = np.random.normal(0, 0.001)
                    interval_price = price * (1 + price_change)
                    prices.append(interval_price)
                
                execution_price = np.mean(prices)
                fill_ratio = 1.0
                
            else:  # VWAP
                # Volume-weighted average price
                n_intervals = self.config.vwap_window
                interval_size = size / n_intervals
                
                prices = []
                volumes = []
                for i in range(n_intervals):
                    # Simulate price and volume
                    price_change = np.random.normal(0, 0.001)
                    volume = np.random.lognormal(10, 1)
                    
                    interval_price = price * (1 + price_change)
                    prices.append(interval_price)
                    volumes.append(volume)
                
                # Calculate VWAP
                execution_price = np.average(prices, weights=volumes)
                fill_ratio = 1.0
            
            return execution_price, fill_ratio
            
        except Exception as e:
            logger.error(f"Error simulating TWAP/VWAP: {str(e)}")
            return price, 0
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate execution metrics.
        
        Returns:
            Dictionary of execution metrics
        """
        try:
            if not self.trades:
                return {}
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            total_volume = trades_df['size'].sum()
            total_value = (trades_df['size'] * trades_df['price']).sum()
            
            # Fill metrics
            avg_fill_ratio = trades_df['fill_ratio'].mean()
            dark_pool_ratio = trades_df['dark_pool'].mean()
            
            # Price metrics
            price_improvement = (
                trades_df[trades_df['side'] == 'buy']['price'].mean() -
                trades_df[trades_df['side'] == 'sell']['price'].mean()
            ) / trades_df['price'].mean()
            
            # Cost metrics
            slippage = (
                trades_df['price'] - trades_df['price'].shift(1)
            ).abs().mean() / trades_df['price'].mean()
            
            # Time metrics
            trade_duration = (
                trades_df['timestamp'].max() - trades_df['timestamp'].min()
            ).total_seconds() / 3600  # hours
            
            return {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_value': total_value,
                'avg_fill_ratio': avg_fill_ratio,
                'dark_pool_ratio': dark_pool_ratio,
                'price_improvement': price_improvement,
                'slippage': slippage,
                'trade_duration': trade_duration
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

class ExecutionEngine:
    """Handles portfolio execution and simulation."""
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        initial_cash: float = 1e6
    ):
        """
        Initialize execution engine.
        
        Args:
            config: Execution configuration
            initial_cash: Initial cash balance
        """
        self.config = config or ExecutionConfig()
        self.order_manager = OrderManager(config, initial_cash)
        self.metrics = {}
    
    def calculate_market_impact(
        self,
        symbol: str,
        size: float,
        price: float,
        volume: float,
        time_of_day: float,
        order_book_depth: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate market impact with advanced models.
        
        Args:
            symbol: Asset symbol
            size: Order size
            price: Current price
            volume: Daily volume
            time_of_day: Time of day (0-1)
            order_book_depth: Order book depth
        
        Returns:
            Market impact in basis points
        """
        try:
            # Base impact
            if self.config.slippage_model == 'sqrt':
                # Square root model
                impact = self.config.market_impact_coef * np.sqrt(size / volume)
            else:
                # Linear model
                impact = self.config.market_impact_coef * (size / volume)
            
            # Time of day adjustment
            if time_of_day < 0.1 or time_of_day > 0.9:  # Market open/close
                impact *= 1.5
            
            # Order book depth adjustment
            if order_book_depth is not None:
                depth_ratio = size / order_book_depth.get('total', volume)
                impact *= (1 + depth_ratio)
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            return 0
    
    def simulate_execution(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        order_book_depth: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Simulate portfolio execution.
        
        Args:
            signals: Trading signals
            prices: Asset prices
            volumes: Trading volumes
            order_book_depth: Order book depth
        
        Returns:
            Execution results
        """
        try:
            results = []
            
            for timestamp in signals.index:
                # Get current market data
                current_prices = prices.loc[timestamp]
                current_volumes = volumes.loc[timestamp]
                current_depth = order_book_depth.get(timestamp, {}) if order_book_depth else None
                
                # Calculate time of day
                time_of_day = timestamp.hour / 24 + timestamp.minute / (24 * 60)
                
                # Process signals
                for symbol in signals.columns:
                    signal = signals.loc[timestamp, symbol]
                    
                    if signal != 0:  # Active signal
                        # Calculate order size
                        size = self.order_manager.calculate_order_size(
                            symbol,
                            abs(signal),
                            current_prices[symbol],
                            volatility=0.1,  # Placeholder
                            time_of_day=time_of_day
                        )
                        
                        # Calculate market impact
                        impact = self.calculate_market_impact(
                            symbol,
                            size,
                            current_prices[symbol],
                            current_volumes[symbol],
                            time_of_day,
                            current_depth.get(symbol, {}) if current_depth else None
                        )
                        
                        # Execute order
                        side = 'buy' if signal > 0 else 'sell'
                        order_result = self.order_manager.execute_order(
                            symbol,
                            side,
                            size,
                            current_prices[symbol] * (1 + impact),
                            order_type='market',
                            dark_pool=np.random.random() < self.config.dark_pool_ratio
                        )
                        
                        # Record result
                        if order_result['status'] == 'filled':
                            results.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'signal': signal,
                                'size': size,
                                'price': order_result['trade']['price'],
                                'impact': impact,
                                'fill_ratio': order_result['trade']['fill_ratio'],
                                'dark_pool': order_result['trade']['dark_pool']
                            })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate metrics
            self.metrics = self.order_manager.calculate_metrics()
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error simulating execution: {str(e)}")
            return pd.DataFrame()

def main():
    """Main execution function."""
    try:
        # Initialize execution engine
        engine = ExecutionEngine()
        
        # Load data
        signals_path = config.paths.processed_data_dir / "trading_signals.parquet"
        prices_path = config.paths.processed_data_dir / "processed_data_20240315_1200.parquet"
        
        signals = pd.read_parquet(signals_path)
        data = pd.read_parquet(prices_path)
        
        # Prepare data
        prices = data.pivot(columns='symbol', values='close')
        volumes = data.pivot(columns='symbol', values='volume')
        
        # Simulate execution
        results = engine.simulate_execution(signals, prices, volumes)
        
        # Save results
        results.to_parquet(
            config.paths.processed_data_dir / "execution_results.parquet"
        )
        pd.Series(engine.metrics).to_frame('value').to_parquet(
            config.paths.processed_data_dir / "execution_metrics.parquet"
        )
        
        logger.info("Execution simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Execution simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()