
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

export const MarketOverview = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([
    { symbol: 'SPY', name: 'S&P 500 ETF', price: 428.50, change: 2.35, changePercent: 0.55 },
    { symbol: 'QQQ', name: 'NASDAQ 100 ETF', price: 368.20, change: -1.80, changePercent: -0.49 },
    { symbol: 'IWM', name: 'Russell 2000 ETF', price: 198.75, change: 0.92, changePercent: 0.46 },
    { symbol: 'VIX', name: 'Volatility Index', price: 18.42, change: -0.68, changePercent: -3.56 },
    { symbol: 'DXY', name: 'US Dollar Index', price: 103.85, change: 0.15, changePercent: 0.14 },
    { symbol: 'GLD', name: 'Gold ETF', price: 189.20, change: -0.85, changePercent: -0.45 },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => prev.map(item => ({
        ...item,
        price: item.price + (Math.random() - 0.5) * 0.5,
        change: item.change + (Math.random() - 0.5) * 0.1,
        changePercent: item.changePercent + (Math.random() - 0.5) * 0.05,
      })));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="terminal-panel h-full flex items-center">
      <div className="flex space-x-8 overflow-x-auto p-4 w-full">
        {marketData.map((item) => (
          <div key={item.symbol} className="flex-shrink-0 min-w-40">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-mono font-semibold text-terminal-cyan">
                  {item.symbol}
                </div>
                <div className="text-xs text-terminal-muted truncate">
                  {item.name}
                </div>
              </div>
              <div className="text-right">
                <div className="financial-number font-semibold">
                  ${item.price.toFixed(2)}
                </div>
                <div className={`flex items-center text-xs ${
                  item.change >= 0 ? 'status-positive' : 'status-negative'
                }`}>
                  {item.change >= 0 ? (
                    <TrendingUp className="w-3 h-3 mr-1" />
                  ) : (
                    <TrendingDown className="w-3 h-3 mr-1" />
                  )}
                  <span className="financial-number">
                    {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)} 
                    ({item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
