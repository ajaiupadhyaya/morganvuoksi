"""
Backtesting engine for evaluating trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 1000000)
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 10 bps
        self.slippage = config.get('slippage', 0.0002)  # 2 bps
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        self.position_size = config.get('position_size', 1.0)  # Full position
        self.stop_loss = config.get('stop_loss', 0.02)  # 2% stop loss
        self.take_profit = config.get('take_profit', 0.04)  # 4% take profit
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> Dict:
        """Run backtest on historical data with signals."""
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['position'] = 0
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0
        portfolio['total'] = self.initial_capital
        portfolio['returns'] = 0
        
        # Track trades
        trades = []
        current_position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            # Get current price with slippage
            price = data['Close'].iloc[i]
            if signals.iloc[i] > 0:  # Buy signal
                price *= (1 + self.slippage)
            elif signals.iloc[i] < 0:  # Sell signal
                price *= (1 - self.slippage)
            
            # Update position
            if signals.iloc[i] != 0 and signals.iloc[i] != current_position:
                # Close existing position
                if current_position != 0:
                    exit_price = price
                    pnl = (exit_price - entry_price) * current_position
                    portfolio.loc[data.index[i], 'cash'] += pnl
                    trades.append({
                        'entry_date': data.index[i-1],
                        'exit_date': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': current_position,
                        'pnl': pnl
                    })
                
                # Open new position
                current_position = signals.iloc[i]
                entry_price = price
                portfolio.loc[data.index[i], 'position'] = current_position
                
                # Apply transaction costs
                cost = abs(current_position) * price * self.transaction_cost
                portfolio.loc[data.index[i], 'cash'] -= cost
            
            # Update portfolio
            portfolio.loc[data.index[i], 'holdings'] = current_position * price
            portfolio.loc[data.index[i], 'total'] = (
                portfolio.loc[data.index[i], 'cash'] +
                portfolio.loc[data.index[i], 'holdings']
            )
            portfolio.loc[data.index[i], 'returns'] = (
                portfolio.loc[data.index[i], 'total'] /
                portfolio.loc[data.index[i-1], 'total'] - 1
            )
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio)
        
        return {
            'portfolio': portfolio,
            'trades': pd.DataFrame(trades),
            'metrics': metrics
        }
    
    def calculate_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        returns = portfolio['returns']
        
        # Basic metrics
        total_return = (portfolio['total'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Win rate
        trades = portfolio[portfolio['position'] != portfolio['position'].shift(1)]
        winning_trades = trades[trades['returns'] > 0]
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades)
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Plot equity curve
        ax1 = fig.add_subplot(gs[0, :])
        results['portfolio']['total'].plot(ax=ax1, label='Portfolio Value')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        
        # Plot drawdown
        ax2 = fig.add_subplot(gs[1, :])
        drawdown = (results['portfolio']['total'] / results['portfolio']['total'].expanding().max() - 1)
        drawdown.plot(ax=ax2, label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        
        # Plot returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        sns.histplot(results['portfolio']['returns'], ax=ax3, bins=50)
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Frequency')
        
        # Plot rolling metrics
        ax4 = fig.add_subplot(gs[2, 1])
        rolling_sharpe = results['portfolio']['returns'].rolling(252).mean() / results['portfolio']['returns'].rolling(252).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=ax4, label='Rolling Sharpe')
        ax4.set_title('Rolling Sharpe Ratio')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def generate_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """Generate detailed backtest report."""
        metrics = results['metrics']
        trades = results['trades']
        
        report = f"""
Backtest Report
==============

Performance Metrics
-----------------
Total Return: {metrics['total_return']:.2%}
Annual Return: {metrics['annual_return']:.2%}
Volatility: {metrics['volatility']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2%}
Win Rate: {metrics['win_rate']:.2%}
Profit Factor: {metrics['profit_factor']:.2f}
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Number of Trades: {metrics['num_trades']}

Trade Statistics
--------------
Average Trade Return: {trades['pnl'].mean():.2f}
Best Trade: {trades['pnl'].max():.2f}
Worst Trade: {trades['pnl'].min():.2f}
Average Trade Duration: {(trades['exit_date'] - trades['entry_date']).mean().days} days
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report 
