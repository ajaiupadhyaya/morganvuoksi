
import React, { useState, useEffect } from 'react';
import { Wallet, TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

interface PortfolioData {
  totalValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  cash: number;
  positions: number;
}

export const PortfolioSummary = () => {
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    totalValue: 1250000,
    dailyPnL: 12450,
    dailyPnLPercent: 1.01,
    totalPnL: 185000,
    totalPnLPercent: 17.35,
    cash: 125000,
    positions: 12
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setPortfolio(prev => ({
        ...prev,
        totalValue: prev.totalValue + (Math.random() - 0.5) * 1000,
        dailyPnL: prev.dailyPnL + (Math.random() - 0.5) * 500,
        dailyPnLPercent: prev.dailyPnLPercent + (Math.random() - 0.5) * 0.1,
        totalPnL: prev.totalPnL + (Math.random() - 0.5) * 200,
        totalPnLPercent: prev.totalPnLPercent + (Math.random() - 0.5) * 0.05,
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3 flex items-center space-x-2">
        <Wallet className="w-4 h-4 text-terminal-orange" />
        <h2 className="text-sm font-semibold text-terminal-orange">PORTFOLIO</h2>
      </div>
      
      <div className="flex-1 p-3 space-y-4">
        {/* Total Value */}
        <div>
          <div className="text-xs text-terminal-muted mb-1">Total Value</div>
          <div className="financial-number text-xl font-bold text-terminal-text">
            ${portfolio.totalValue.toLocaleString()}
          </div>
        </div>
        
        {/* Daily P&L */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-terminal-muted mb-1">Daily P&L</div>
            <div className={`flex items-center space-x-1 ${
              portfolio.dailyPnL >= 0 ? 'status-positive' : 'status-negative'
            }`}>
              {portfolio.dailyPnL >= 0 ? (
                <TrendingUp className="w-3 h-3" />
              ) : (
                <TrendingDown className="w-3 h-3" />
              )}
              <div className="financial-number text-sm font-semibold">
                {portfolio.dailyPnL >= 0 ? '+' : ''}${portfolio.dailyPnL.toLocaleString()}
              </div>
            </div>
            <div className={`financial-number text-xs ${
              portfolio.dailyPnLPercent >= 0 ? 'status-positive' : 'status-negative'
            }`}>
              ({portfolio.dailyPnLPercent >= 0 ? '+' : ''}{portfolio.dailyPnLPercent.toFixed(2)}%)
            </div>
          </div>
          
          <div>
            <div className="text-xs text-terminal-muted mb-1">Total P&L</div>
            <div className={`flex items-center space-x-1 ${
              portfolio.totalPnL >= 0 ? 'status-positive' : 'status-negative'
            }`}>
              {portfolio.totalPnL >= 0 ? (
                <TrendingUp className="w-3 h-3" />
              ) : (
                <TrendingDown className="w-3 h-3" />
              )}
              <div className="financial-number text-sm font-semibold">
                {portfolio.totalPnL >= 0 ? '+' : ''}${portfolio.totalPnL.toLocaleString()}
              </div>
            </div>
            <div className={`financial-number text-xs ${
              portfolio.totalPnLPercent >= 0 ? 'status-positive' : 'status-negative'
            }`}>
              ({portfolio.totalPnLPercent >= 0 ? '+' : ''}{portfolio.totalPnLPercent.toFixed(2)}%)
            </div>
          </div>
        </div>
        
        {/* Additional Metrics */}
        <div className="space-y-3 pt-2 border-t border-terminal-border/50">
          <div className="flex justify-between items-center">
            <span className="text-xs text-terminal-muted">Available Cash</span>
            <span className="financial-number text-sm text-terminal-cyan">
              ${portfolio.cash.toLocaleString()}
            </span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-xs text-terminal-muted">Open Positions</span>
            <span className="financial-number text-sm text-terminal-text">
              {portfolio.positions}
            </span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-xs text-terminal-muted">Buying Power</span>
            <span className="financial-number text-sm text-terminal-green">
              ${(portfolio.cash * 4).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
