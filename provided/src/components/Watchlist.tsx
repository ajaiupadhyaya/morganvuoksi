
import React, { useState, useEffect } from 'react';
import { Star, TrendingUp, TrendingDown } from 'lucide-react';

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  isWatched: boolean;
}

interface WatchlistProps {
  selectedSymbol: string;
  onSymbolSelect: (symbol: string) => void;
}

export const Watchlist = ({ selectedSymbol, onSymbolSelect }: WatchlistProps) => {
  const [stocks, setStocks] = useState<Stock[]>([
    { symbol: 'AAPL', name: 'Apple Inc.', price: 180.25, change: 2.45, changePercent: 1.38, isWatched: true },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 2750.80, change: -12.30, changePercent: -0.45, isWatched: true },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 340.60, change: 5.20, changePercent: 1.55, isWatched: true },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 245.15, change: -8.75, changePercent: -3.45, isWatched: true },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 875.30, change: 15.60, changePercent: 1.81, isWatched: true },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 3180.50, change: 25.80, changePercent: 0.82, isWatched: false },
    { symbol: 'META', name: 'Meta Platforms', price: 485.20, change: -3.40, changePercent: -0.70, isWatched: false },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setStocks(prev => prev.map(stock => ({
        ...stock,
        price: stock.price + (Math.random() - 0.5) * 2,
        change: stock.change + (Math.random() - 0.5) * 0.5,
        changePercent: stock.changePercent + (Math.random() - 0.5) * 0.1,
      })));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const toggleWatch = (symbol: string) => {
    setStocks(prev => prev.map(stock => 
      stock.symbol === symbol ? { ...stock, isWatched: !stock.isWatched } : stock
    ));
  };

  const watchedStocks = stocks.filter(stock => stock.isWatched);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <h2 className="text-sm font-semibold text-terminal-orange">WATCHLIST</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th className="text-left">Symbol</th>
              <th className="text-right">Price</th>
              <th className="text-right">Change</th>
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {watchedStocks.map((stock) => (
              <tr 
                key={stock.symbol}
                className={`cursor-pointer transition-colors ${
                  selectedSymbol === stock.symbol ? 'bg-terminal-orange/20' : ''
                }`}
                onClick={() => onSymbolSelect(stock.symbol)}
              >
                <td>
                  <div>
                    <div className="font-mono font-semibold text-terminal-cyan">
                      {stock.symbol}
                    </div>
                    <div className="text-xs text-terminal-muted truncate">
                      {stock.name}
                    </div>
                  </div>
                </td>
                <td className="text-right">
                  <div className="financial-number font-medium">
                    ${stock.price.toFixed(2)}
                  </div>
                </td>
                <td className="text-right">
                  <div className={`flex items-center justify-end text-xs ${
                    stock.change >= 0 ? 'status-positive' : 'status-negative'
                  }`}>
                    {stock.change >= 0 ? (
                      <TrendingUp className="w-3 h-3 mr-1" />
                    ) : (
                      <TrendingDown className="w-3 h-3 mr-1" />
                    )}
                    <div>
                      <div className="financial-number">
                        {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                      </div>
                      <div className="financial-number">
                        ({stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                </td>
                <td>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleWatch(stock.symbol);
                    }}
                    className="hover:text-terminal-orange transition-colors"
                  >
                    <Star className={`w-3 h-3 ${stock.isWatched ? 'fill-current text-terminal-orange' : 'text-terminal-muted'}`} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
