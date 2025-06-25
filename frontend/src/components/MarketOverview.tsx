"use client";

import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Clock, Users } from 'lucide-react';
import { TerminalData } from '@/types';

interface MarketOverviewProps {
  data: TerminalData;
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ data }) => {
  const { symbol, watchlist } = data;
  const isPositiveChange = symbol.change_pct >= 0;

  return (
    <div className="h-full p-6 space-y-6">
      {/* Main Symbol Overview */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            {symbol.ticker} - {symbol.name}
          </h3>
          <div className="flex items-center gap-2">
            <span className="status-indicator status-live"></span>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">LIVE QUOTES</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Price */}
          <div className="metric-card">
            <div className="metric-label">LAST PRICE</div>
            <div className="metric-value">${symbol.price.toFixed(2)}</div>
          </div>

          {/* Change */}
          <div className="metric-card">
            <div className="metric-label">CHANGE</div>
            <div className="metric-value">
              {isPositiveChange ? '+' : ''}{symbol.change_val.toFixed(2)}
            </div>
            <div className={`metric-change ${isPositiveChange ? 'positive-change' : 'negative-change'}`}>
              {isPositiveChange ? <TrendingUp className="w-3 h-3 inline mr-1" /> : <TrendingDown className="w-3 h-3 inline mr-1" />}
              {isPositiveChange ? '+' : ''}{symbol.change_pct.toFixed(2)}%
            </div>
          </div>

          {/* Volume */}
          <div className="metric-card">
            <div className="metric-label">VOLUME</div>
            <div className="metric-value text-lg">{symbol.volume}</div>
          </div>

          {/* Market Cap */}
          <div className="metric-card">
            <div className="metric-label">MARKET CAP</div>
            <div className="metric-value text-lg">{symbol.market_cap}</div>
          </div>

          {/* Time */}
          <div className="metric-card">
            <div className="metric-label">MARKET TIME</div>
            <div className="metric-value text-lg flex items-center gap-2">
              <Clock className="w-4 h-4" />
              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
      </div>

      {/* Market Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Watchlist */}
        <div className="bloomberg-card">
          <div className="bloomberg-card-header">
            <h3 className="bloomberg-card-title flex items-center gap-2">
              <Users className="w-4 h-4" />
              WATCHLIST
            </h3>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
              {watchlist.length} SYMBOLS
            </span>
          </div>

          <div className="space-y-2">
            {watchlist.map((item, index) => {
              const itemIsPositive = item.change_pct >= 0;
              return (
                <div 
                  key={index}
                  className="flex items-center justify-between p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)] hover:bg-[var(--bloomberg-hover)] transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-mono font-bold text-sm">{item.ticker}</span>
                  </div>
                  <div className="flex items-center gap-4 text-right">
                    <span className="font-mono font-semibold">${item.price.toFixed(2)}</span>
                    <span className={`font-mono text-sm ${itemIsPositive ? 'positive-change' : 'negative-change'}`}>
                      {itemIsPositive ? <TrendingUp className="w-3 h-3 inline mr-1" /> : <TrendingDown className="w-3 h-3 inline mr-1" />}
                      {itemIsPositive ? '+' : ''}{item.change_pct.toFixed(2)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Market Indices */}
        <div className="bloomberg-card">
          <div className="bloomberg-card-header">
            <h3 className="bloomberg-card-title flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              MAJOR INDICES
            </h3>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">REAL-TIME</span>
          </div>

          <div className="space-y-2">
            {[
              { symbol: 'SPX', name: 'S&P 500', price: 4567.89, change: 12.34, changePct: 0.27 },
              { symbol: 'DJI', name: 'Dow Jones', price: 34567.89, change: -45.67, changePct: -0.13 },
              { symbol: 'IXIC', name: 'NASDAQ', price: 14234.56, change: 78.90, changePct: 0.56 },
              { symbol: 'RUT', name: 'Russell 2000', price: 2123.45, change: -8.76, changePct: -0.41 },
            ].map((index, idx) => {
              const indexIsPositive = index.changePct >= 0;
              return (
                <div 
                  key={idx}
                  className="flex items-center justify-between p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)] hover:bg-[var(--bloomberg-hover)] transition-colors"
                >
                  <div>
                    <div className="font-mono font-bold text-sm">{index.symbol}</div>
                    <div className="text-xs text-[var(--bloomberg-text-secondary)]">{index.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-semibold">{index.price.toLocaleString()}</div>
                    <div className={`font-mono text-xs ${indexIsPositive ? 'positive-change' : 'negative-change'}`}>
                      {indexIsPositive ? <TrendingUp className="w-3 h-3 inline mr-1" /> : <TrendingDown className="w-3 h-3 inline mr-1" />}
                      {indexIsPositive ? '+' : ''}{index.change.toFixed(2)} ({indexIsPositive ? '+' : ''}{index.changePct.toFixed(2)}%)
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Market Status */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <DollarSign className="w-4 h-4" />
            MARKET STATUS
          </h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center justify-between p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
            <span className="text-sm text-[var(--bloomberg-text-secondary)]">Market Status</span>
            <div className="flex items-center gap-2">
              <span className="status-indicator status-live"></span>
              <span className="font-mono font-bold text-[var(--bloomberg-terminal-green)]">OPEN</span>
            </div>
          </div>
          
          <div className="flex items-center justify-between p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
            <span className="text-sm text-[var(--bloomberg-text-secondary)]">Session</span>
            <span className="font-mono font-bold">REGULAR</span>
          </div>
          
          <div className="flex items-center justify-between p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
            <span className="text-sm text-[var(--bloomberg-text-secondary)]">Next Close</span>
            <span className="font-mono font-bold">16:00 EST</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketOverview;