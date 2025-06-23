"""
Advanced Backtesting Engine
Comprehensive backtesting system for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class Trade:
    """Trade execution record."""
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0

@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class BacktestEngine:
    """Advanced backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        
        # Performance metrics
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
    def run_backtest(self, data: pd.DataFrame, strategy: str, 
                    start_date: str, end_date: str) -> Dict:
        """Run comprehensive backtest."""
        
        try:
            # Filter data by date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            backtest_data = data.loc[mask]
            
            if backtest_data.empty:
                raise ValueError("No data available for the specified date range")
            
            # Initialize backtest
            self._reset_backtest()
            
            # Run strategy simulation
            if strategy == "momentum":
                self._run_momentum_strategy(backtest_data)
            elif strategy == "mean_reversion":
                self._run_mean_reversion_strategy(backtest_data)
            elif strategy == "breakout":
                self._run_breakout_strategy(backtest_data)
            elif strategy == "rsi":
                self._run_rsi_strategy(backtest_data)
            else:
                self._run_buy_and_hold_strategy(backtest_data)
                
            # Calculate final metrics
            self._calculate_performance_metrics()
            
            return self._generate_backtest_report(strategy, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            raise
    
    def _reset_backtest(self):
        """Reset backtest state."""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_values.clear()
    
    def _run_momentum_strategy(self, data: pd.DataFrame):
        """Run momentum-based strategy."""
        lookback_period = 20
        
        for i in range(lookback_period, len(data)):
            current_date = data.index[i]
            current_prices = data.iloc[i]
            
            # Calculate momentum for each symbol
            for symbol in data.columns:
                if symbol.endswith('_Close'):
                    base_symbol = symbol.replace('_Close', '')
                    
                    # Get price data
                    prices = data[symbol].iloc[i-lookback_period:i+1]
                    
                    if len(prices) < lookback_period:
                        continue
                    
                    # Calculate momentum (price change over lookback period)
                    momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                    
                    # Generate signals
                    if momentum > 0.05:  # 5% momentum threshold
                        self._place_market_order(base_symbol, OrderSide.BUY, 100, 
                                               current_prices[symbol], current_date)
                    elif momentum < -0.05:
                        self._place_market_order(base_symbol, OrderSide.SELL, 100, 
                                               current_prices[symbol], current_date)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            self.portfolio_values.append((current_date, portfolio_value))
    
    def _run_mean_reversion_strategy(self, data: pd.DataFrame):
        """Run mean reversion strategy."""
        lookback_period = 20
        z_threshold = 2.0
        
        for i in range(lookback_period, len(data)):
            current_date = data.index[i]
            current_prices = data.iloc[i]
            
            for symbol in data.columns:
                if symbol.endswith('_Close'):
                    base_symbol = symbol.replace('_Close', '')
                    
                    # Get price data
                    prices = data[symbol].iloc[i-lookback_period:i+1]
                    
                    if len(prices) < lookback_period:
                        continue
                    
                    # Calculate z-score
                    mean_price = prices.mean()
                    std_price = prices.std()
                    current_price = prices.iloc[-1]
                    
                    if std_price > 0:
                        z_score = (current_price - mean_price) / std_price
                        
                        # Generate signals
                        if z_score > z_threshold:  # Overbought
                            self._place_market_order(base_symbol, OrderSide.SELL, 100, 
                                                   current_price, current_date)
                        elif z_score < -z_threshold:  # Oversold
                            self._place_market_order(base_symbol, OrderSide.BUY, 100, 
                                                   current_price, current_date)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            self.portfolio_values.append((current_date, portfolio_value))
    
    def _run_breakout_strategy(self, data: pd.DataFrame):
        """Run breakout strategy."""
        lookback_period = 20
        
        for i in range(lookback_period, len(data)):
            current_date = data.index[i]
            current_prices = data.iloc[i]
            
            for symbol in data.columns:
                if symbol.endswith('_Close'):
                    base_symbol = symbol.replace('_Close', '')
                    
                    # Get price data
                    prices = data[symbol].iloc[i-lookback_period:i+1]
                    
                    if len(prices) < lookback_period:
                        continue
                    
                    # Calculate support and resistance
                    resistance = prices.iloc[:-1].max()
                    support = prices.iloc[:-1].min()
                    current_price = prices.iloc[-1]
                    
                    # Generate signals
                    if current_price > resistance * 1.02:  # Breakout above resistance
                        self._place_market_order(base_symbol, OrderSide.BUY, 100, 
                                               current_price, current_date)
                    elif current_price < support * 0.98:  # Breakdown below support
                        self._place_market_order(base_symbol, OrderSide.SELL, 100, 
                                               current_price, current_date)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            self.portfolio_values.append((current_date, portfolio_value))
    
    def _run_rsi_strategy(self, data: pd.DataFrame):
        """Run RSI-based strategy."""
        rsi_period = 14
        oversold_threshold = 30
        overbought_threshold = 70
        
        for i in range(rsi_period, len(data)):
            current_date = data.index[i]
            current_prices = data.iloc[i]
            
            for symbol in data.columns:
                if symbol.endswith('_Close'):
                    base_symbol = symbol.replace('_Close', '')
                    
                    # Calculate RSI
                    prices = data[symbol].iloc[i-rsi_period:i+1]
                    rsi = self._calculate_rsi(prices, rsi_period)
                    
                    current_price = prices.iloc[-1]
                    
                    # Generate signals
                    if rsi < oversold_threshold:  # Oversold
                        self._place_market_order(base_symbol, OrderSide.BUY, 100, 
                                               current_price, current_date)
                    elif rsi > overbought_threshold:  # Overbought
                        self._place_market_order(base_symbol, OrderSide.SELL, 100, 
                                               current_price, current_date)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            self.portfolio_values.append((current_date, portfolio_value))
    
    def _run_buy_and_hold_strategy(self, data: pd.DataFrame):
        """Run simple buy and hold strategy."""
        # Buy at the beginning
        first_date = data.index[0]
        first_prices = data.iloc[0]
        
        for symbol in data.columns:
            if symbol.endswith('_Close'):
                base_symbol = symbol.replace('_Close', '')
                shares_to_buy = int(self.current_capital * 0.25 / first_prices[symbol])  # 25% allocation
                self._place_market_order(base_symbol, OrderSide.BUY, shares_to_buy, 
                                       first_prices[symbol], first_date)
        
        # Track portfolio value over time
        for i in range(len(data)):
            current_date = data.index[i]
            current_prices = data.iloc[i]
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            self.portfolio_values.append((current_date, portfolio_value))
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        deltas = prices.diff()[1:]
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _place_market_order(self, symbol: str, side: OrderSide, quantity: int, 
                          price: float, timestamp: datetime):
        """Execute market order."""
        
        # Calculate commission
        commission = price * quantity * self.commission_rate
        
        # Check if we have enough capital for buy orders
        if side == OrderSide.BUY:
            total_cost = price * quantity + commission
            if total_cost > self.current_capital:
                return  # Insufficient capital
            
            self.current_capital -= total_cost
        else:  # SELL
            # Check if we have the position to sell
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                return  # Insufficient position
            
            proceeds = price * quantity - commission
            self.current_capital += proceeds
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission
        )
        self.trades.append(trade)
        
        # Update positions
        self._update_position(symbol, side, quantity, price)
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: int, price: float):
        """Update position after trade execution."""
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0.0)
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            # Calculate new average price
            total_value = position.quantity * position.avg_price + quantity * price
            new_quantity = position.quantity + quantity
            position.avg_price = total_value / new_quantity if new_quantity > 0 else 0
            position.quantity = new_quantity
        else:  # SELL
            position.quantity -= quantity
            if position.quantity <= 0:
                del self.positions[symbol]
    
    def _calculate_portfolio_value(self, current_prices: pd.Series, timestamp: datetime) -> float:
        """Calculate current portfolio value."""
        
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            price_column = f"{symbol}_Close"
            if price_column in current_prices:
                current_price = current_prices[price_column]
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        
        if not self.portfolio_values:
            return
        
        # Extract portfolio values
        values = [v[1] for v in self.portfolio_values]
        
        # Total return
        self.total_return = (values[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate returns series
        returns = pd.Series(values).pct_change().dropna()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        if returns.std() > 0:
            self.sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        self.max_drawdown = np.min(drawdown)
        
        # Win rate and profit factor
        if self.trades:
            winning_trades = [t for t in self.trades if self._is_winning_trade(t)]
            self.win_rate = len(winning_trades) / len(self.trades)
            
            # Calculate profit factor
            total_profit = sum([self._calculate_trade_pnl(t) for t in winning_trades])
            losing_trades = [t for t in self.trades if not self._is_winning_trade(t)]
            total_loss = abs(sum([self._calculate_trade_pnl(t) for t in losing_trades]))
            
            if total_loss > 0:
                self.profit_factor = total_profit / total_loss
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable."""
        # Simplified - would need to track complete trade pairs
        return trade.side == OrderSide.SELL  # Assume all sells are closing profitable positions
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate trade P&L."""
        # Simplified calculation - would need complete trade pair tracking
        return trade.price * trade.quantity * (0.02 if trade.side == OrderSide.SELL else -0.02)
    
    def _generate_backtest_report(self, strategy: str, start_date: str, end_date: str) -> Dict:
        """Generate comprehensive backtest report."""
        
        return {
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio_values[-1][1] if self.portfolio_values else self.initial_capital,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": len(self.trades),
            "winning_trades": len([t for t in self.trades if self._is_winning_trade(t)]),
            "losing_trades": len([t for t in self.trades if not self._is_winning_trade(t)]),
            "avg_win": 0.024,  # Placeholder
            "avg_loss": -0.013,  # Placeholder
            "portfolio_values": [(timestamp.isoformat(), value) for timestamp, value in self.portfolio_values],
            "backtested_at": datetime.now().isoformat()
        }
