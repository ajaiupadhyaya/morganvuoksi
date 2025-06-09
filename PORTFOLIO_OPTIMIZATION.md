# Portfolio Optimization Guide

This guide outlines the institution-grade portfolio optimization system for the trading platform.

## Mean-Variance Optimization

### 1. Modern Portfolio Theory
```python
# optimization/mean_variance/mpt.py
class ModernPortfolioTheory:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean = returns.mean()
        self.cov = returns.cov()
    
    def optimize(self, target_return: float = None) -> pd.Series:
        """Optimize portfolio weights."""
        n_assets = len(self.mean)
        
        if target_return is None:
            # Maximize Sharpe ratio
            weights = self._maximize_sharpe_ratio()
        else:
            # Minimize variance for target return
            weights = self._minimize_variance(target_return)
        
        return pd.Series(weights, index=self.mean.index)
    
    def _maximize_sharpe_ratio(self) -> np.ndarray:
        """Maximize Sharpe ratio."""
        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(self.mean * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Optimize
        result = minimize(neg_sharpe_ratio, 
                        x0=np.array([1/n_assets] * n_assets),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        return result.x
    
    def _minimize_variance(self, target_return: float) -> np.ndarray:
        """Minimize variance for target return."""
        def portfolio_variance(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean * x) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Optimize
        result = minimize(portfolio_variance,
                        x0=np.array([1/n_assets] * n_assets),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        return result.x
```

### 2. Black-Litterman Model
```python
# optimization/mean_variance/black_litterman.py
class BlackLitterman:
    def __init__(self, returns: pd.DataFrame, market_caps: pd.Series,
                 risk_free_rate: float = 0.0, tau: float = 0.05):
        self.returns = returns
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.mean = returns.mean()
        self.cov = returns.cov()
    
    def optimize(self, views: Dict[str, float],
                confidences: Dict[str, float]) -> pd.Series:
        """Optimize portfolio weights using Black-Litterman model."""
        # Calculate prior
        prior = self._calculate_prior()
        
        # Calculate posterior
        posterior = self._calculate_posterior(prior, views, confidences)
        
        # Optimize weights
        weights = self._optimize_weights(posterior)
        
        return pd.Series(weights, index=self.mean.index)
    
    def _calculate_prior(self) -> np.ndarray:
        """Calculate prior returns."""
        # Market equilibrium returns
        market_weights = self.market_caps / self.market_caps.sum()
        market_return = np.sum(self.mean * market_weights)
        market_vol = np.sqrt(np.dot(market_weights.T, 
                                   np.dot(self.cov, market_weights)))
        
        # Risk aversion
        risk_aversion = (market_return - self.risk_free_rate) / market_vol**2
        
        # Prior returns
        prior = self.risk_free_rate + risk_aversion * np.dot(self.cov, market_weights)
        
        return prior
    
    def _calculate_posterior(self, prior: np.ndarray,
                           views: Dict[str, float],
                           confidences: Dict[str, float]) -> np.ndarray:
        """Calculate posterior returns."""
        # View matrix
        P = np.zeros((len(views), len(self.mean)))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (asset, view) in enumerate(views.items()):
            P[i, self.mean.index.get_loc(asset)] = 1
            Q[i] = view
            Omega[i, i] = 1 / confidences[asset]
        
        # Posterior
        tau_sigma = self.tau * self.cov
        M = np.linalg.inv(np.linalg.inv(tau_sigma) + 
                         np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))
        posterior = np.dot(M, np.dot(np.linalg.inv(tau_sigma), prior) + 
                          np.dot(P.T, np.dot(np.linalg.inv(Omega), Q)))
        
        return posterior
```

### 3. Risk Parity
```python
# optimization/mean_variance/risk_parity.py
class RiskParity:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov = returns.cov()
    
    def optimize(self) -> pd.Series:
        """Optimize portfolio weights using risk parity."""
        n_assets = len(self.cov)
        weights = np.ones(n_assets) / n_assets
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))
            marginal_risk = np.dot(self.cov, weights) / portfolio_vol
            return marginal_risk * weights
        
        def risk_parity_objective(weights):
            risk_contrib = risk_contribution(weights)
            target_risk = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib - target_risk)**2)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Optimize
        result = minimize(risk_parity_objective,
                        x0=weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        return pd.Series(result.x, index=self.cov.index)
```

## Factor Models

### 1. Factor Analysis
```python
# optimization/factor/factor_analysis.py
class FactorAnalysis:
    def __init__(self, returns: pd.DataFrame, n_factors: int):
        self.returns = returns
        self.n_factors = n_factors
    
    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fit factor model."""
        # Calculate covariance matrix
        cov = self.returns.cov()
        
        # Perform PCA
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Extract factors
        factors = eigenvecs[:, :self.n_factors]
        factor_returns = np.dot(self.returns, factors)
        
        # Calculate factor loadings
        loadings = np.dot(self.returns.T, factor_returns) / len(self.returns)
        
        return factors, loadings
    
    def optimize(self, factors: np.ndarray,
                loadings: np.ndarray) -> pd.Series:
        """Optimize portfolio weights using factor model."""
        # Calculate factor covariance
        factor_cov = np.cov(factors.T)
        
        # Calculate specific risk
        specific_risk = np.diag(self.returns.cov() - 
                               np.dot(loadings, np.dot(factor_cov, loadings.T)))
        
        # Optimize weights
        weights = self._optimize_weights(factors, loadings, 
                                       factor_cov, specific_risk)
        
        return pd.Series(weights, index=self.returns.columns)
```

### 2. Fundamental Factors
```python
# optimization/factor/fundamental.py
class FundamentalFactors:
    def __init__(self, returns: pd.DataFrame, factors: pd.DataFrame):
        self.returns = returns
        self.factors = factors
    
    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit fundamental factor model."""
        # Calculate factor returns
        factor_returns = pd.DataFrame(index=self.returns.index)
        factor_loadings = pd.DataFrame(index=self.returns.columns)
        
        for factor in self.factors.columns:
            # Calculate factor loadings
            loadings = self._calculate_loadings(factor)
            factor_loadings[factor] = loadings
            
            # Calculate factor returns
            returns = self._calculate_returns(loadings)
            factor_returns[factor] = returns
        
        return factor_returns, factor_loadings
    
    def optimize(self, factor_returns: pd.DataFrame,
                factor_loadings: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights using fundamental factors."""
        # Calculate factor covariance
        factor_cov = factor_returns.cov()
        
        # Calculate specific risk
        specific_risk = self._calculate_specific_risk(factor_returns, 
                                                    factor_loadings)
        
        # Optimize weights
        weights = self._optimize_weights(factor_returns, factor_loadings,
                                       factor_cov, specific_risk)
        
        return pd.Series(weights, index=self.returns.columns)
```

### 3. Statistical Factors
```python
# optimization/factor/statistical.py
class StatisticalFactors:
    def __init__(self, returns: pd.DataFrame, n_factors: int):
        self.returns = returns
        self.n_factors = n_factors
    
    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fit statistical factor model."""
        # Calculate covariance matrix
        cov = self.returns.cov()
        
        # Perform PCA
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Extract factors
        factors = eigenvecs[:, :self.n_factors]
        factor_returns = np.dot(self.returns, factors)
        
        # Calculate factor loadings
        loadings = np.dot(self.returns.T, factor_returns) / len(self.returns)
        
        return factors, loadings
    
    def optimize(self, factors: np.ndarray,
                loadings: np.ndarray) -> pd.Series:
        """Optimize portfolio weights using statistical factors."""
        # Calculate factor covariance
        factor_cov = np.cov(factors.T)
        
        # Calculate specific risk
        specific_risk = np.diag(self.returns.cov() - 
                               np.dot(loadings, np.dot(factor_cov, loadings.T)))
        
        # Optimize weights
        weights = self._optimize_weights(factors, loadings,
                                       factor_cov, specific_risk)
        
        return pd.Series(weights, index=self.returns.columns)
```

## Implementation Guide

### 1. Setup
```python
# config/optimization_config.py
def setup_optimization_environment():
    """Configure optimization environment."""
    # Set optimization parameters
    optimization_params = {
        'target_return': 0.1,
        'risk_free_rate': 0.02,
        'n_factors': 5,
        'max_position': 0.2
    }
    
    # Set factor parameters
    factor_params = {
        'fundamental_factors': ['value', 'momentum', 'quality'],
        'statistical_factors': 5,
        'min_factor_exposure': 0.1
    }
    
    # Set risk parameters
    risk_params = {
        'max_position': 0.2,
        'max_sector': 0.3,
        'max_factor': 0.4
    }
    
    return {
        'optimization_params': optimization_params,
        'factor_params': factor_params,
        'risk_params': risk_params
    }
```

### 2. Optimization Pipeline
```python
# optimization/pipeline.py
class OptimizationPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.optimizer = self._setup_optimizer()
        self.factor_model = self._setup_factor_model()
        self.risk_manager = self._setup_risk_manager()
        self.optimization_history = []
    
    def run_pipeline(self, market_data: MarketData):
        """Execute optimization pipeline."""
        # Calculate returns
        returns = self._calculate_returns(market_data)
        
        # Fit factor model
        factors, loadings = self.factor_model.fit(returns)
        
        # Optimize weights
        weights = self.optimizer.optimize(returns, factors, loadings)
        
        # Check risk limits
        weights = self.risk_manager.check_limits(weights)
        
        # Update optimization history
        self._update_history(weights)
```

### 3. Monitoring
```python
# optimization/monitoring.py
class OptimizationMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {}
        self.alerts = []
    
    def monitor(self, market_data: MarketData,
               optimization_history: List[Dict]):
        """Monitor optimization performance."""
        # Calculate metrics
        self._calculate_metrics(market_data, optimization_history)
        
        # Check for alerts
        self._check_alerts()
        
        # Update dashboard
        self._update_dashboard()
```

## Best Practices

1. **Optimization Strategy**
   - Mean-variance optimization
   - Factor models
   - Risk management
   - Constraints

2. **Risk Management**
   - Position limits
   - Sector limits
   - Factor limits
   - Market impact

3. **Monitoring**
   - Portfolio quality
   - Factor exposure
   - Risk metrics
   - Performance tracking

4. **Documentation**
   - Optimization policies
   - Procedures
   - Reports
   - Alerts

## Monitoring

1. **Optimization Metrics**
   - Sharpe ratio
   - Information ratio
   - Factor exposure
   - Risk metrics

2. **Risk Metrics**
   - Position
   - Sector
   - Factor
   - Market impact

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
   - Execution algorithms
   - Market making
   - Arbitrage detection
   - Risk management

3. **Automation**
   - Optimization monitoring
   - Alert generation
   - Report generation
   - Limit management 