# Arbitrage Detection Guide

This guide outlines the institution-grade arbitrage detection system for the trading platform.

## Statistical Arbitrage

### 1. Pairs Trading
```python
# arbitrage/statistical/pairs_trading.py
class PairsTrader:
    def __init__(self, lookback: int, entry_threshold: float,
                 exit_threshold: float):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.pairs = {}
    
    def find_pairs(self, prices: Dict[str, pd.Series]) -> List[Tuple[str, str]]:
        """Find cointegrated pairs."""
        pairs = []
        for i in range(len(prices)):
            for j in range(i+1, len(prices)):
                series1 = prices.iloc[:, i]
                series2 = prices.iloc[:, j]
                if self._check_cointegration(series1, series2):
                    pairs.append((prices.columns[i], prices.columns[j]))
        return pairs
    
    def calculate_spread(self, pair: Tuple[str, str],
                        prices: Dict[str, pd.Series]) -> pd.Series:
        """Calculate spread between pair."""
        series1 = prices[pair[0]]
        series2 = prices[pair[1]]
        
        # Calculate hedge ratio
        hedge_ratio = self._calculate_hedge_ratio(series1, series2)
        
        # Calculate spread
        spread = series1 - hedge_ratio * series2
        
        return spread
    
    def generate_signals(self, spread: pd.Series) -> pd.Series:
        """Generate trading signals."""
        # Calculate z-score
        z_score = (spread - spread.rolling(self.lookback).mean()) / \
                  spread.rolling(self.lookback).std()
        
        # Generate signals
        signals = pd.Series(0, index=spread.index)
        signals[z_score > self.entry_threshold] = -1  # Short spread
        signals[z_score < -self.entry_threshold] = 1  # Long spread
        signals[abs(z_score) < self.exit_threshold] = 0  # Exit position
        
        return signals
```

### 2. Mean Reversion
```python
# arbitrage/statistical/mean_reversion.py
class MeanReversionTrader:
    def __init__(self, lookback: int, entry_threshold: float,
                 exit_threshold: float):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """Calculate z-score of prices."""
        return (prices - prices.rolling(self.lookback).mean()) / \
               prices.rolling(self.lookback).std()
    
    def generate_signals(self, zscore: pd.Series) -> pd.Series:
        """Generate trading signals."""
        signals = pd.Series(0, index=zscore.index)
        signals[zscore > self.entry_threshold] = -1  # Short
        signals[zscore < -self.entry_threshold] = 1  # Long
        signals[abs(zscore) < self.exit_threshold] = 0  # Exit
        
        return signals
    
    def calculate_position_size(self, zscore: pd.Series,
                              volatility: pd.Series) -> pd.Series:
        """Calculate position size."""
        # Base position size
        position_size = -zscore
        
        # Adjust for volatility
        position_size = position_size / volatility
        
        # Normalize
        position_size = position_size / position_size.abs().max()
        
        return position_size
```

### 3. Momentum
```python
# arbitrage/statistical/momentum.py
class MomentumTrader:
    def __init__(self, lookback: int, entry_threshold: float,
                 exit_threshold: float):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate momentum signal."""
        return prices.pct_change(self.lookback)
    
    def generate_signals(self, momentum: pd.Series) -> pd.Series:
        """Generate trading signals."""
        signals = pd.Series(0, index=momentum.index)
        signals[momentum > self.entry_threshold] = 1  # Long
        signals[momentum < -self.entry_threshold] = -1  # Short
        signals[abs(momentum) < self.exit_threshold] = 0  # Exit
        
        return signals
    
    def calculate_position_size(self, momentum: pd.Series,
                              volatility: pd.Series) -> pd.Series:
        """Calculate position size."""
        # Base position size
        position_size = momentum
        
        # Adjust for volatility
        position_size = position_size / volatility
        
        # Normalize
        position_size = position_size / position_size.abs().max()
        
        return position_size
```

## Cross-Exchange Arbitrage

### 1. Triangular Arbitrage
```python
# arbitrage/cross_exchange/triangular.py
class TriangularArbitrage:
    def __init__(self, exchanges: List[Exchange]):
        self.exchanges = exchanges
    
    def find_opportunities(self, prices: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Find triangular arbitrage opportunities."""
        opportunities = []
        
        for exchange in self.exchanges:
            # Get all possible triangles
            triangles = self._get_triangles(exchange)
            
            for triangle in triangles:
                # Calculate arbitrage profit
                profit = self._calculate_profit(triangle, prices[exchange])
                
                if profit > 0:
                    opportunities.append({
                        'exchange': exchange,
                        'triangle': triangle,
                        'profit': profit
                    })
        
        return opportunities
    
    def _calculate_profit(self, triangle: List[str],
                         prices: Dict[str, float]) -> float:
        """Calculate arbitrage profit."""
        # Start with 1 unit of base currency
        amount = 1.0
        
        # Execute trades
        for i in range(len(triangle)-1):
            pair = f"{triangle[i]}/{triangle[i+1]}"
            amount *= prices[pair]
        
        # Return to base currency
        pair = f"{triangle[-1]}/{triangle[0]}"
        amount *= prices[pair]
        
        return amount - 1.0
```

### 2. Statistical Arbitrage
```python
# arbitrage/cross_exchange/statistical.py
class CrossExchangeArbitrage:
    def __init__(self, exchanges: List[Exchange],
                 lookback: int, threshold: float):
        self.exchanges = exchanges
        self.lookback = lookback
        self.threshold = threshold
    
    def find_opportunities(self, prices: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Find cross-exchange arbitrage opportunities."""
        opportunities = []
        
        for exchange1 in self.exchanges:
            for exchange2 in self.exchanges:
                if exchange1 != exchange2:
                    # Calculate price differences
                    differences = self._calculate_differences(
                        prices[exchange1],
                        prices[exchange2]
                    )
                    
                    # Find opportunities
                    for pair, diff in differences.items():
                        if abs(diff) > self.threshold:
                            opportunities.append({
                                'pair': pair,
                                'exchange1': exchange1,
                                'exchange2': exchange2,
                                'difference': diff
                            })
        
        return opportunities
    
    def _calculate_differences(self, prices1: Dict[str, float],
                             prices2: Dict[str, float]) -> Dict[str, float]:
        """Calculate price differences between exchanges."""
        differences = {}
        
        for pair in prices1:
            if pair in prices2:
                diff = (prices1[pair] - prices2[pair]) / prices1[pair]
                differences[pair] = diff
        
        return differences
```

### 3. Market Making
```python
# arbitrage/cross_exchange/market_making.py
class CrossExchangeMarketMaker:
    def __init__(self, exchanges: List[Exchange],
                 spread_threshold: float):
        self.exchanges = exchanges
        self.spread_threshold = spread_threshold
    
    def find_opportunities(self, orderbooks: Dict[str, Dict[str, OrderBook]]) -> List[Dict]:
        """Find cross-exchange market making opportunities."""
        opportunities = []
        
        for exchange1 in self.exchanges:
            for exchange2 in self.exchanges:
                if exchange1 != exchange2:
                    # Calculate spreads
                    spreads = self._calculate_spreads(
                        orderbooks[exchange1],
                        orderbooks[exchange2]
                    )
                    
                    # Find opportunities
                    for pair, spread in spreads.items():
                        if spread > self.spread_threshold:
                            opportunities.append({
                                'pair': pair,
                                'exchange1': exchange1,
                                'exchange2': exchange2,
                                'spread': spread
                            })
        
        return opportunities
    
    def _calculate_spreads(self, orderbook1: Dict[str, OrderBook],
                          orderbook2: Dict[str, OrderBook]) -> Dict[str, float]:
        """Calculate spreads between exchanges."""
        spreads = {}
        
        for pair in orderbook1:
            if pair in orderbook2:
                # Get best bid and ask
                bid1 = orderbook1[pair].get_best_bid()
                ask1 = orderbook1[pair].get_best_ask()
                bid2 = orderbook2[pair].get_best_bid()
                ask2 = orderbook2[pair].get_best_ask()
                
                # Calculate spread
                spread = min(ask1 - bid2, ask2 - bid1)
                spreads[pair] = spread
        
        return spreads
```

## Implementation Guide

### 1. Setup
```python
# config/arbitrage_config.py
def setup_arbitrage_environment():
    """Configure arbitrage environment."""
    # Set statistical arbitrage parameters
    stat_arb_params = {
        'lookback': 20,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    }
    
    # Set cross-exchange parameters
    cross_exchange_params = {
        'spread_threshold': 0.001,
        'min_profit': 0.0005,
        'max_position': 1000000
    }
    
    # Set risk parameters
    risk_params = {
        'max_position': 1000000,
        'max_drawdown': 50000,
        'max_volatility': 0.5
    }
    
    return {
        'stat_arb_params': stat_arb_params,
        'cross_exchange_params': cross_exchange_params,
        'risk_params': risk_params
    }
```

### 2. Arbitrage Pipeline
```python
# arbitrage/pipeline.py
class ArbitragePipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.stat_arb = self._setup_statistical_arbitrage()
        self.cross_exchange = self._setup_cross_exchange_arbitrage()
        self.risk_manager = self._setup_risk_manager()
        self.execution_history = []
    
    def run_pipeline(self, market_data: MarketData):
        """Execute arbitrage pipeline."""
        # Find opportunities
        opportunities = self._find_opportunities(market_data)
        
        # Check risk limits
        opportunities = self._check_risk_limits(opportunities)
        
        # Execute trades
        self._execute_trades(opportunities)
        
        # Update execution history
        self._update_history(opportunities)
```

### 3. Monitoring
```python
# arbitrage/monitoring.py
class ArbitrageMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {}
        self.alerts = []
    
    def monitor(self, market_data: MarketData, 
               execution_history: List[Dict]):
        """Monitor arbitrage performance."""
        # Calculate metrics
        self._calculate_metrics(market_data, execution_history)
        
        # Check for alerts
        self._check_alerts()
        
        # Update dashboard
        self._update_dashboard()
```

## Best Practices

1. **Arbitrage Strategy**
   - Statistical arbitrage
   - Cross-exchange arbitrage
   - Market making
   - Risk management

2. **Risk Management**
   - Position limits
   - PnL limits
   - Volatility limits
   - Market impact

3. **Monitoring**
   - Opportunity quality
   - Execution quality
   - PnL
   - Risk metrics

4. **Documentation**
   - Arbitrage policies
   - Procedures
   - Reports
   - Alerts

## Monitoring

1. **Arbitrage Metrics**
   - Opportunity count
   - Execution rate
   - PnL
   - Market impact

2. **Risk Metrics**
   - Position
   - PnL
   - Volatility
   - Drawdown

3. **System Metrics**
   - Latency
   - Throughput
   - Memory usage
   - CPU utilization

## Future Enhancements

1. **Advanced Models**
   - Machine learning
   - Deep learning
   - Reinforcement learning
   - Causal inference

2. **Integration Points**
   - Portfolio optimization
   - Execution algorithms
   - Market making
   - Risk management

3. **Automation**
   - Opportunity monitoring
   - Alert generation
   - Report generation
   - Limit management 