"use client";

import React from 'react';
import { PieChart, TrendingUp, TrendingDown, Target, Shield, DollarSign, BarChart3 } from 'lucide-react';
import { TerminalData } from '@/types';

interface PortfolioSummaryProps {
  data: TerminalData;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = () => {
  // Mock portfolio data - in real implementation, this would come from your portfolio API
  const portfolioData = {
    totalValue: 1247563.89,
    dayChange: 12847.52,
    dayChangePct: 1.04,
    totalReturn: 187432.15,
    totalReturnPct: 17.73,
    cash: 45678.90,
    holdings: [
      { symbol: 'AAPL', shares: 150, avgCost: 145.67, currentPrice: 175.23, value: 26284.50, weight: 21.1 },
      { symbol: 'GOOGL', shares: 25, avgCost: 2456.78, currentPrice: 2687.34, value: 67183.50, weight: 15.8 },
      { symbol: 'MSFT', shares: 100, avgCost: 287.34, currentPrice: 323.45, value: 32345.00, weight: 12.3 },
      { symbol: 'TSLA', shares: 50, avgCost: 234.56, currentPrice: 267.89, value: 13394.50, weight: 8.9 },
      { symbol: 'NVDA', shares: 75, avgCost: 189.23, currentPrice: 434.56, value: 32592.00, weight: 11.2 }
    ]
  };

  const isDayPositive = portfolioData.dayChangePct >= 0;
  const isTotalPositive = portfolioData.totalReturnPct >= 0;

  return (
    <div className="h-full p-6 space-y-6">
      {/* Portfolio Overview */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <PieChart className="w-4 h-4" />
            PORTFOLIO OVERVIEW
          </h3>
          <div className="flex items-center gap-2">
            <span className="status-indicator status-live"></span>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">REAL-TIME</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Total Value */}
          <div className="metric-card">
            <div className="metric-label">TOTAL VALUE</div>
            <div className="metric-value">${portfolioData.totalValue.toLocaleString()}</div>
          </div>

          {/* Day Change */}
          <div className="metric-card">
            <div className="metric-label">DAY CHANGE</div>
            <div className="metric-value">
              {isDayPositive ? '+' : ''}${portfolioData.dayChange.toLocaleString()}
            </div>
            <div className={`metric-change ${isDayPositive ? 'positive-change' : 'negative-change'}`}>
              {isDayPositive ? <TrendingUp className="w-3 h-3 inline mr-1" /> : <TrendingDown className="w-3 h-3 inline mr-1" />}
              {isDayPositive ? '+' : ''}{portfolioData.dayChangePct.toFixed(2)}%
            </div>
          </div>

          {/* Total Return */}
          <div className="metric-card">
            <div className="metric-label">TOTAL RETURN</div>
            <div className="metric-value">
              {isTotalPositive ? '+' : ''}${portfolioData.totalReturn.toLocaleString()}
            </div>
            <div className={`metric-change ${isTotalPositive ? 'positive-change' : 'negative-change'}`}>
              {isTotalPositive ? <TrendingUp className="w-3 h-3 inline mr-1" /> : <TrendingDown className="w-3 h-3 inline mr-1" />}
              {isTotalPositive ? '+' : ''}{portfolioData.totalReturnPct.toFixed(2)}%
            </div>
          </div>

          {/* Cash */}
          <div className="metric-card">
            <div className="metric-label">CASH</div>
            <div className="metric-value">${portfolioData.cash.toLocaleString()}</div>
            <div className="metric-change text-[var(--bloomberg-text-secondary)]">
              <DollarSign className="w-3 h-3 inline mr-1" />
              {((portfolioData.cash / portfolioData.totalValue) * 100).toFixed(1)}%
            </div>
          </div>

          {/* Holdings Count */}
          <div className="metric-card">
            <div className="metric-label">POSITIONS</div>
            <div className="metric-value">{portfolioData.holdings.length}</div>
            <div className="metric-change text-[var(--bloomberg-text-secondary)]">
              <Target className="w-3 h-3 inline mr-1" />
              ACTIVE
            </div>
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            PORTFOLIO HOLDINGS
          </h3>
          <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
            {portfolioData.holdings.length} POSITIONS
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="bloomberg-table">
            <thead>
              <tr>
                <th>SYMBOL</th>
                <th className="text-right">SHARES</th>
                <th className="text-right">AVG COST</th>
                <th className="text-right">CURRENT</th>
                <th className="text-right">P&L</th>
                <th className="text-right">VALUE</th>
                <th className="text-right">WEIGHT</th>
              </tr>
            </thead>
            <tbody>
              {portfolioData.holdings.map((holding, index) => {
                const pnl = (holding.currentPrice - holding.avgCost) * holding.shares;
                const pnlPct = ((holding.currentPrice - holding.avgCost) / holding.avgCost) * 100;
                const isPnlPositive = pnl >= 0;

                return (
                  <tr key={index}>
                    <td className="font-bold text-[var(--bloomberg-text-primary)]">{holding.symbol}</td>
                    <td className="text-right">{holding.shares.toLocaleString()}</td>
                    <td className="text-right">${holding.avgCost.toFixed(2)}</td>
                    <td className="text-right">${holding.currentPrice.toFixed(2)}</td>
                    <td className={`text-right ${isPnlPositive ? 'text-[var(--gains-color)]' : 'text-[var(--losses-color)]'}`}>
                      {isPnlPositive ? '+' : ''}${pnl.toLocaleString()} 
                      <br />
                      <span className="text-xs">
                        ({isPnlPositive ? '+' : ''}{pnlPct.toFixed(2)}%)
                      </span>
                    </td>
                    <td className="text-right font-semibold">${holding.value.toLocaleString()}</td>
                    <td className="text-right">{holding.weight.toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Portfolio Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Asset Allocation */}
        <div className="bloomberg-card">
          <div className="bloomberg-card-header">
            <h3 className="bloomberg-card-title">ASSET ALLOCATION</h3>
          </div>
          
          <div className="space-y-3">
            {portfolioData.holdings.slice(0, 5).map((holding, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ 
                      backgroundColor: `hsl(${(index * 60) % 360}, 60%, 50%)` 
                    }}
                  ></div>
                  <span className="font-mono text-sm">{holding.symbol}</span>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm">{holding.weight.toFixed(1)}%</div>
                  <div className="text-xs text-[var(--bloomberg-text-secondary)]">
                    ${holding.value.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Cash allocation */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-[var(--bloomberg-text-secondary)]"></div>
                <span className="font-mono text-sm">CASH</span>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm">
                  {((portfolioData.cash / portfolioData.totalValue) * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-[var(--bloomberg-text-secondary)]">
                  ${portfolioData.cash.toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bloomberg-card">
          <div className="bloomberg-card-header">
            <h3 className="bloomberg-card-title flex items-center gap-2">
              <Shield className="w-4 h-4" />
              RISK METRICS
            </h3>
          </div>
          
          <div className="space-y-4">
            {[
              { label: 'Beta', value: '1.23', description: 'vs S&P 500' },
              { label: 'Sharpe Ratio', value: '1.87', description: '12M trailing' },
              { label: 'Max Drawdown', value: '-8.4%', description: 'Last 12M' },
              { label: 'Volatility', value: '18.7%', description: 'Annualized' },
              { label: 'Correlation', value: '0.82', description: 'vs Market' }
            ].map((metric, index) => (
              <div key={index} className="flex items-center justify-between p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
                <div>
                  <div className="font-mono font-semibold text-sm">{metric.label}</div>
                  <div className="text-xs text-[var(--bloomberg-text-secondary)]">{metric.description}</div>
                </div>
                <div className="font-mono font-bold text-lg">{metric.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;