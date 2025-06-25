
import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, TrendingDown, Target } from 'lucide-react';

interface RiskMetrics {
  portfolioValue: number;
  dailyVar: number;
  maxDrawdown: number;
  sharpeRatio: number;
  beta: number;
  volatility: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export const RiskManager = () => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    portfolioValue: 1250000,
    dailyVar: -18500,
    maxDrawdown: -8.5,
    sharpeRatio: 1.45,
    beta: 1.12,
    volatility: 18.5,
    riskLevel: 'medium'
  });

  const [alerts, setAlerts] = useState([
    'Portfolio concentration risk: 45% in Tech sector',
    'VaR limit exceeded by $2,500',
    'High correlation detected in top 3 positions'
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setRiskMetrics(prev => ({
        ...prev,
        dailyVar: prev.dailyVar + (Math.random() - 0.5) * 1000,
        volatility: prev.volatility + (Math.random() - 0.5) * 0.5,
        sharpeRatio: prev.sharpeRatio + (Math.random() - 0.5) * 0.1,
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-terminal-green';
      case 'medium': return 'text-terminal-amber';
      case 'high': return 'text-terminal-red';
      default: return 'text-terminal-muted';
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">RISK MANAGER</h2>
          </div>
          <div className={`text-xs font-medium ${getRiskColor(riskMetrics.riskLevel)}`}>
            {riskMetrics.riskLevel.toUpperCase()} RISK
          </div>
        </div>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-y-auto">
        {/* Key Risk Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-xs text-terminal-muted mb-1">Daily VaR (95%)</div>
            <div className="text-terminal-red financial-number text-sm font-semibold">
              ${riskMetrics.dailyVar.toLocaleString()}
            </div>
          </div>
          
          <div>
            <div className="text-xs text-terminal-muted mb-1">Max Drawdown</div>
            <div className="text-terminal-red financial-number text-sm font-semibold">
              {riskMetrics.maxDrawdown}%
            </div>
          </div>
          
          <div>
            <div className="text-xs text-terminal-muted mb-1">Sharpe Ratio</div>
            <div className="text-terminal-cyan financial-number text-sm font-semibold">
              {riskMetrics.sharpeRatio.toFixed(2)}
            </div>
          </div>
          
          <div>
            <div className="text-xs text-terminal-muted mb-1">Portfolio Beta</div>
            <div className="text-terminal-text financial-number text-sm font-semibold">
              {riskMetrics.beta.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Volatility Gauge */}
        <div>
          <div className="text-xs text-terminal-muted mb-2">Volatility: {riskMetrics.volatility.toFixed(1)}%</div>
          <div className="w-full bg-terminal-border rounded-full h-2">
            <div 
              className="bg-terminal-amber h-2 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(riskMetrics.volatility * 2, 100)}%` }}
            />
          </div>
        </div>

        {/* Risk Alerts */}
        <div className="border-t border-terminal-border/50 pt-3">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="w-3 h-3 text-terminal-red" />
            <span className="text-xs font-medium text-terminal-red">ALERTS</span>
          </div>
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <div key={index} className="text-xs text-terminal-muted bg-terminal-bg p-2 rounded border-l-2 border-terminal-red">
                {alert}
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-2 pt-2">
          <button className="flex items-center justify-center space-x-1 py-1 text-xs bg-terminal-border hover:bg-terminal-orange/20 rounded transition-colors">
            <Target className="w-3 h-3" />
            <span>HEDGE</span>
          </button>
          <button className="flex items-center justify-center space-x-1 py-1 text-xs bg-terminal-border hover:bg-terminal-red/20 rounded transition-colors">
            <TrendingDown className="w-3 h-3" />
            <span>REDUCE</span>
          </button>
        </div>
      </div>
    </div>
  );
};
