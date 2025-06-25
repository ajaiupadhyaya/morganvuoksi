"use client";

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Zap, Globe, DollarSign, BarChart3, Clock, Users } from 'lucide-react';
import { TerminalData } from '@/types';

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  lastUpdate: string;
  sector: string;
}

interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  status: 'up' | 'down' | 'neutral';
}

interface MarketOverviewProps {
  data: TerminalData;
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ data }) => {
  const { symbol, watchlist } = data;
  const isPositiveChange = symbol.change_pct >= 0;

  const [marketData, setMarketData] = useState<MarketData[]>([
    { 
      symbol: 'SPY', 
      name: 'SPDR S&P 500 ETF', 
      price: 428.50, 
      change: 2.35, 
      changePercent: 0.55,
      volume: 85420000,
      high: 430.25,
      low: 426.80,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'ETF'
    },
    { 
      symbol: 'QQQ', 
      name: 'Invesco QQQ Trust', 
      price: 368.20, 
      change: -1.80, 
      changePercent: -0.49,
      volume: 52100000,
      high: 371.45,
      low: 367.10,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'ETF'
    },
    { 
      symbol: 'IWM', 
      name: 'iShares Russell 2000', 
      price: 198.75, 
      change: 0.92, 
      changePercent: 0.46,
      volume: 29800000,
      high: 199.80,
      low: 197.65,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'ETF'
    },
    { 
      symbol: 'VIX', 
      name: 'CBOE Volatility Index', 
      price: 18.42, 
      change: -0.68, 
      changePercent: -3.56,
      volume: 0,
      high: 19.85,
      low: 18.12,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'INDEX'
    },
    { 
      symbol: 'DXY', 
      name: 'US Dollar Index', 
      price: 103.85, 
      change: 0.15, 
      changePercent: 0.14,
      volume: 0,
      high: 104.20,
      low: 103.45,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'CURRENCY'
    },
    { 
      symbol: 'GLD', 
      name: 'SPDR Gold Trust', 
      price: 189.20, 
      change: -0.85, 
      changePercent: -0.45,
      volume: 12500000,
      high: 190.50,
      low: 188.80,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'COMMODITY'
    },
    { 
      symbol: 'TSLA', 
      name: 'Tesla Inc', 
      price: 248.75, 
      change: 8.32, 
      changePercent: 3.46,
      volume: 95400000,
      high: 252.10,
      low: 245.20,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'AUTOMOTIVE'
    },
    { 
      symbol: 'AAPL', 
      name: 'Apple Inc', 
      price: 184.92, 
      change: -1.25, 
      changePercent: -0.67,
      volume: 67800000,
      high: 186.50,
      low: 183.80,
      lastUpdate: new Date().toLocaleTimeString(),
      sector: 'TECHNOLOGY'
    },
  ]);

  const [majorIndices, setMajorIndices] = useState<MarketIndex[]>([
    { symbol: 'SPX', name: 'S&P 500', value: 4521.23, change: 12.45, changePercent: 0.28, status: 'up' },
    { symbol: 'NDX', name: 'NASDAQ 100', value: 15245.67, change: -8.23, changePercent: -0.05, status: 'down' },
    { symbol: 'DJI', name: 'DOW JONES', value: 34892.10, change: 156.78, changePercent: 0.45, status: 'up' },
    { symbol: 'RUT', name: 'RUSSELL 2000', value: 1987.45, change: 8.92, changePercent: 0.45, status: 'up' },
  ]);

  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre' | 'after'>('open');

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simulate real-time market data updates
      setMarketData(prev => prev.map(item => {
        const priceChange = (Math.random() - 0.5) * 0.5;
        const newPrice = Math.max(0.01, item.price + priceChange);
        const newChange = item.change + (Math.random() - 0.5) * 0.1;
        
        return {
          ...item,
          price: newPrice,
          change: newChange,
          changePercent: (newChange / (newPrice - newChange)) * 100,
          lastUpdate: new Date().toLocaleTimeString(),
        };
      }));

      setMajorIndices(prev => prev.map(index => {
        const change = (Math.random() - 0.5) * 5;
        const newValue = index.value + change;
        const newChange = index.change + (Math.random() - 0.5) * 0.5;
        
        return {
          ...index,
          value: newValue,
          change: newChange,
          changePercent: (newChange / (newValue - newChange)) * 100,
          status: newChange >= 0 ? 'up' : 'down'
        };
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const getStatusColor = (change: number) => {
    if (change > 0) return 'status-positive';
    if (change < 0) return 'status-negative';
    return 'status-neutral';
  };

  return (
    <div className="terminal-panel h-full flex flex-col bg-terminal-bg border-2 border-terminal-border">
      {/* Professional Header Bar */}
      <div className="flex items-center justify-between px-3 py-1 border-b-2 border-terminal-orange bg-gradient-to-r from-terminal-panel to-terminal-bg">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Globe className="w-4 h-4 text-terminal-orange terminal-pulse" />
            <span className="text-terminal-orange font-mono font-bold text-sm uppercase tracking-wider">
              GLOBAL MARKETS
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              marketStatus === 'open' ? 'bg-terminal-green' : 'bg-terminal-red'
            } terminal-pulse`} />
            <span className="text-terminal-cyan font-mono text-xs uppercase">
              {marketStatus === 'open' ? 'MARKET OPEN' : 'MARKET CLOSED'}
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-6">
          {/* Major Indices Quick View */}
          {majorIndices.slice(0, 3).map((index) => (
            <div key={index.symbol} className="flex items-center space-x-2 text-xs">
              <span className="text-terminal-cyan font-mono font-bold">{index.symbol}:</span>
              <span className="text-terminal-text font-mono tabular-nums">
                {index.value.toLocaleString()}
              </span>
              <span className={`font-mono tabular-nums ${getStatusColor(index.change)}`}>
                {index.change >= 0 ? '+' : ''}{index.change.toFixed(2)}
              </span>
            </div>
          ))}
          
          <div className="flex items-center space-x-2 text-xs text-terminal-muted">
            <Zap className="w-3 h-3" />
            <span className="font-mono">{currentTime.toLocaleTimeString()} EST</span>
          </div>
        </div>
      </div>

      {/* Professional Scrolling Ticker */}
      <div className="market-ticker relative overflow-hidden h-16 flex items-center">
        <div className="ticker-scroll flex items-center space-x-8 whitespace-nowrap">
          {[...marketData, ...marketData].map((item, index) => (
            <div key={`${item.symbol}-${index}`} className="flex-shrink-0 px-4 py-2 bg-terminal-panel/50 border border-terminal-border/50">
              <div className="flex items-center space-x-3">
                {/* Symbol and Icon */}
                <div className="flex items-center space-x-2">
                  {item.sector === 'ETF' && <BarChart3 className="w-3 h-3 text-terminal-orange" />}
                  {item.sector === 'CURRENCY' && <DollarSign className="w-3 h-3 text-terminal-amber" />}
                  {item.sector === 'INDEX' && <AlertTriangle className="w-3 h-3 text-terminal-red" />}
                  <span className="font-mono font-bold text-terminal-cyan text-sm">
                    {item.symbol}
                  </span>
                </div>
                
                {/* Price Data */}
                <div className="flex items-center space-x-2">
                  <span className="financial-number text-terminal-text font-bold">
                    ${item.price.toFixed(2)}
                  </span>
                  <div className={`flex items-center space-x-1 ${getStatusColor(item.change)}`}>
                    {item.change >= 0 ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    <span className="financial-number text-xs">
                      {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}
                    </span>
                    <span className="financial-number text-xs">
                      ({item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%)
                    </span>
                  </div>
                </div>
                
                {/* Volume and Range */}
                <div className="flex items-center space-x-2 text-xs text-terminal-muted">
                  {item.volume > 0 && (
                    <span className="font-mono">VOL: {formatVolume(item.volume)}</span>
                  )}
                  <span className="font-mono">H: {item.high.toFixed(2)}</span>
                  <span className="font-mono">L: {item.low.toFixed(2)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Market Status Bar */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-terminal-border bg-terminal-panel/30 text-xs">
        <div className="flex items-center space-x-4">
          <span className="text-terminal-cyan font-mono font-bold">MARKET STATUS:</span>
          <span className={`font-mono ${
            marketStatus === 'open' ? 'text-terminal-green' : 'text-terminal-red'
          }`}>
            {marketStatus.toUpperCase()}
          </span>
          <span className="text-terminal-muted font-mono">
            LAST UPDATE: {currentTime.toLocaleTimeString()}
          </span>
        </div>
        
        <div className="flex items-center space-x-4">
          <span className="text-terminal-muted font-mono">DATA FEED:</span>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-terminal-green rounded-full terminal-pulse" />
            <span className="text-terminal-green font-mono">LIVE</span>
          </div>
          <span className="text-terminal-muted font-mono">LATENCY: 0.8ms</span>
        </div>
      </div>
    </div>
  );
};

export default MarketOverview;