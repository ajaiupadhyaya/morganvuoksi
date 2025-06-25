"""
Trading infrastructure for high-performance trading.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import redis
import zmq
import uvicorn
from fastapi import FastAPI, WebSocket
from ib_insync import IB, Stock, MarketOrder
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import cvxpy as cp
import riskfolio as rp
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ..utils.logging import setup_logger
from ..config import get_config

logger = setup_logger(__name__)

class TradingInfrastructure:
    """Trading infrastructure for high-performance trading."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._setup_infrastructure()
        self._setup_apis()
        self._setup_metrics()
    
    def _setup_infrastructure(self):
        """Setup trading infrastructure."""
        # Initialize Ray for distributed computing
        ray.init(
            address=self.config.get('ray_address', 'auto'),
            namespace=self.config.get('ray_namespace', 'trading')
        )
        
        # Setup Redis for caching
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            password=self.config.get('redis_password')
        )
        
        # Setup ZeroMQ for high-performance messaging
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.config.get('zmq_port', 5555)}")
        
        # Setup FastAPI
        self.app = FastAPI()
        self._setup_routes()
    
    def _setup_apis(self):
        """Setup trading APIs."""
        # Interactive Brokers
        if 'ib' in self.config:
            self.ib = IB()
            self.ib.connect(
                self.config['ib']['host'],
                self.config['ib']['port'],
                clientId=self.config['ib']['client_id']
            )
        
        # Alpaca
        if 'alpaca' in self.config:
            self.alpaca = TradingClient(
                self.config['alpaca']['api_key'],
                self.config['alpaca']['api_secret'],
                paper=True if 'paper' in self.config['alpaca'] and self.config['alpaca']['paper'] else False
            )
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        # Trading metrics
        self.trades_counter = Counter(
            'trades_total',
            'Total number of trades executed'
        )
        self.trade_value = Gauge(
            'trade_value',
            'Value of trades in USD'
        )
        self.trade_latency = Histogram(
            'trade_latency_seconds',
            'Trade execution latency in seconds'
        )
        
        # Performance metrics
        self.portfolio_value = Gauge(
            'portfolio_value',
            'Current portfolio value in USD'
        )
        self.position_count = Gauge(
            'position_count',
            'Number of current positions'
        )
        
        # Risk metrics
        self.var_95 = Gauge(
            'var_95',
            '95% Value at Risk'
        )
        self.var_99 = Gauge(
            'var_99',
            '99% Value at Risk'
        )
        
        # Start Prometheus server
        start_http_server(8000)
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    # Process incoming data
                    response = self._process_websocket_data(data)
                    await websocket.send_text(response)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
    
    async def execute_trade(self, order: Dict) -> Dict:
        """
        Execute trade.
        
        Args:
            order: Order details
            
        Returns:
            Execution results
        """
        try:
            # Start timing
            start_time = datetime.now()
            
            if order['broker'] == 'ib':
                # Create IB order
                contract = Stock(order['symbol'], 'SMART', 'USD')
                ib_order = MarketOrder(
                    order['action'],
                    order['quantity']
                )
                
                # Submit order
                trade = self.ib.placeOrder(contract, ib_order)
                while not trade.isDone():
                    await asyncio.sleep(0.1)
                
                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds()
                
                # Update metrics
                self.trades_counter.inc()
                self.trade_value.set(float(trade.fills[0].execution.price) * float(trade.fills[0].execution.shares))
                self.trade_latency.observe(latency)
                
                return {
                    'status': 'filled',
                    'order_id': trade.order.orderId,
                    'fill_price': trade.fills[0].execution.price,
                    'fill_time': trade.fills[0].execution.time,
                    'latency': latency
                }
            
            elif order['broker'] == 'alpaca':
                # Submit Alpaca order using new alpaca-py API
                order_request = MarketOrderRequest(
                    symbol=order['symbol'],
                    qty=order['quantity'],
                    side=OrderSide.BUY if order['action'] == 'buy' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                response = self.alpaca.submit_order(order_request)
                
                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds()
                
                # Update metrics
                self.trades_counter.inc()
                self.trade_latency.observe(latency)
                
                return {
                    'status': 'submitted',
                    'order_id': response.id,
                    'latency': latency
                }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @ray.remote
    def optimize_portfolio(self, data: pd.DataFrame,
                         constraints: Dict) -> Dict:
        """
        Optimize portfolio using distributed computing.
        
        Args:
            data: Market data
            constraints: Optimization constraints
            
        Returns:
            Optimal portfolio weights
        """
        try:
            # Define optimization function
            def objective(weights):
                returns = np.sum(data.mean() * weights) * 252
                risk = np.sqrt(
                    weights.dot(data.cov() * 252).dot(weights)
                )
                return -returns / risk
            
            # Setup optimization
            from scipy.optimize import minimize
            
            n_assets = len(data.columns)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Run optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return {
                'weights': dict(zip(data.columns, result.x)),
                'sharpe': -result.fun,
                'return': np.sum(data.mean() * result.x) * 252,
                'risk': np.sqrt(
                    result.x.dot(data.cov() * 252).dot(result.x)
                )
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return {}
    
    def run_hyperparameter_tuning(self, model: object,
                                param_space: Dict,
                                num_samples: int = 100) -> Dict:
        """
        Run hyperparameter tuning using Ray Tune.
        
        Args:
            model: Model to tune
            param_space: Parameter space to search
            num_samples: Number of samples to try
            
        Returns:
            Best hyperparameters
        """
        try:
            # Define training function
            def train_model(config):
                # Train model with config
                model.set_params(**config)
                model.fit(X_train, y_train)
                
                # Evaluate model
                score = model.score(X_test, y_test)
                
                # Report results
                tune.report(score=score)
            
            # Setup scheduler
            scheduler = ASHAScheduler(
                metric='score',
                mode='max',
                max_t=100,
                grace_period=10,
                reduction_factor=2
            )
            
            # Run tuning
            analysis = tune.run(
                train_model,
                config=param_space,
                num_samples=num_samples,
                scheduler=scheduler,
                resources_per_trial={'cpu': 1}
            )
            
            return analysis.best_config
            
        except Exception as e:
            logger.error(f"Error running hyperparameter tuning: {str(e)}")
            return {}
    
    def monitor_performance(self) -> Dict:
        """
        Monitor trading performance.
        
        Returns:
            Performance metrics
        """
        try:
            # Get positions
            if hasattr(self, 'ib'):
                positions = self.ib.positions()
            elif hasattr(self, 'alpaca'):
                positions = self.alpaca.list_positions()
            
            # Get account value
            if hasattr(self, 'ib'):
                account = self.ib.accountSummary()
                account_value = float(
                    next(x.value for x in account if x.tag == 'NetLiquidation')
                )
            elif hasattr(self, 'alpaca'):
                account = self.alpaca.get_account()
                account_value = float(account.equity)
            
            # Compute metrics
            metrics = {
                'account_value': account_value,
                'positions': positions,
                'timestamp': pd.Timestamp.now()
            }
            
            # Update Prometheus metrics
            self.portfolio_value.set(account_value)
            self.position_count.set(len(positions))
            
            # Calculate VaR
            if positions:
                returns = pd.DataFrame([p.avgCost for p in positions]).pct_change()
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                self.var_95.set(var_95)
                self.var_99.set(var_99)
            
            # Publish metrics
            self.socket.send_json(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            return {}
    
    def run(self):
        """Run trading infrastructure."""
        uvicorn.run(
            self.app,
            host=self.config.get('host', '0.0.0.0'),
            port=self.config.get('port', 8000)
        )
    
    def close(self):
        """Close all connections."""
        if hasattr(self, 'ib'):
            self.ib.disconnect()
        
        if hasattr(self, 'alpaca'):
            self.alpaca.close()
        
        ray.shutdown()

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Create trading infrastructure
    infrastructure = TradingInfrastructure(config)
    
    # Run infrastructure
    infrastructure.run() 
