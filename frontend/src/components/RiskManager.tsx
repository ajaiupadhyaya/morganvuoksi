"use client";

import React from 'react';
import { AlertTriangle, Shield, TrendingDown, Activity } from 'lucide-react';
import { TerminalData } from '@/types';

interface RiskManagerProps {
  data: TerminalData;
}

const RiskManager: React.FC<RiskManagerProps> = () => {
  const riskMetrics = {
    var95: 45678.90,
    var99: 78923.45,
    maxDrawdown: -12.4,
    volatility: 18.7,
    beta: 1.23,
    sharpe: 1.87,
    alerts: [
      { type: 'warning', message: 'Position concentration exceeds 25% in AAPL', severity: 'medium' },
      { type: 'info', message: 'Portfolio beta within acceptable range', severity: 'low' },
      { type: 'error', message: 'VaR limit breached on TSLA position', severity: 'high' }
    ]
  };

  return (
    <div className="h-full p-6 space-y-6">
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Shield className="w-4 h-4" />
            RISK OVERVIEW
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <div className="metric-label">VAR (95%)</div>
            <div className="metric-value text-[var(--bloomberg-accent-red)]">
              ${riskMetrics.var95.toLocaleString()}
            </div>
            <div className="metric-change">
              <TrendingDown className="w-3 h-3 inline mr-1" />
              1-DAY
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">MAX DRAWDOWN</div>
            <div className="metric-value text-[var(--losses-color)]">
              {riskMetrics.maxDrawdown}%
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">VOLATILITY</div>
            <div className="metric-value">{riskMetrics.volatility}%</div>
            <div className="metric-change text-[var(--bloomberg-text-secondary)]">
              <Activity className="w-3 h-3 inline mr-1" />
              ANNUALIZED
            </div>
          </div>
        </div>
      </div>

      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            RISK ALERTS
          </h3>
        </div>

        <div className="space-y-3">
          {riskMetrics.alerts.map((alert, index) => (
            <div 
              key={index}
              className={`p-4 rounded-md border ${
                alert.severity === 'high' ? 'border-[var(--bloomberg-accent-red)] bg-red-900/20' :
                alert.severity === 'medium' ? 'border-[var(--bloomberg-warning)] bg-yellow-900/20' :
                'border-[var(--bloomberg-blue)] bg-blue-900/20'
              }`}
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                  alert.severity === 'high' ? 'text-[var(--bloomberg-accent-red)]' :
                  alert.severity === 'medium' ? 'text-[var(--bloomberg-warning)]' :
                  'text-[var(--bloomberg-blue)]'
                }`} />
                <div>
                  <div className="font-mono text-sm">{alert.message}</div>
                  <div className="text-xs text-[var(--bloomberg-text-secondary)] mt-1">
                    {alert.severity.toUpperCase()} PRIORITY
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RiskManager;