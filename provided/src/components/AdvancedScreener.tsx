
import React, { useState } from 'react';
import { Search, Filter, TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';

interface ScreenerCriteria {
  marketCap: { min: number; max: number };
  volume: { min: number; max: number };
  priceChange: { min: number; max: number };
  sector: string;
  exchange: string;
}

interface ScreenerResult {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
  pe: number;
}

export const AdvancedScreener = () => {
  const [criteria, setCriteria] = useState<ScreenerCriteria>({
    marketCap: { min: 0, max: 1000000 },
    volume: { min: 100000, max: 10000000 },
    priceChange: { min: -10, max: 10 },
    sector: 'all',
    exchange: 'all'
  });

  const [results, setResults] = useState<ScreenerResult[]>([
    { symbol: 'AAPL', name: 'Apple Inc.', price: 180.25, change: 2.45, changePercent: 1.38, volume: 52000000, marketCap: 2800000, sector: 'Technology', pe: 28.5 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 340.60, change: 5.20, changePercent: 1.55, volume: 28000000, marketCap: 2500000, sector: 'Technology', pe: 32.1 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 2750.80, change: -12.30, changePercent: -0.45, volume: 1200000, marketCap: 1700000, sector: 'Technology', pe: 25.8 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 245.15, change: -8.75, changePercent: -3.45, volume: 95000000, marketCap: 780000, sector: 'Automotive', pe: 65.2 },
  ]);

  const runScreener = () => {
    console.log('Running screener with criteria:', criteria);
    // In a real app, this would call an API
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center space-x-2 mb-2">
          <Filter className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">STOCK SCREENER</h2>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Screening Criteria */}
        <div className="p-3 border-b border-terminal-border/50">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="text-terminal-muted mb-1 block">Market Cap (M)</label>
              <div className="flex space-x-2">
                <input
                  type="number"
                  value={criteria.marketCap.min}
                  onChange={(e) => setCriteria(prev => ({
                    ...prev,
                    marketCap: { ...prev.marketCap, min: Number(e.target.value) }
                  }))}
                  className="terminal-input flex-1 text-xs"
                  placeholder="Min"
                />
                <input
                  type="number"
                  value={criteria.marketCap.max}
                  onChange={(e) => setCriteria(prev => ({
                    ...prev,
                    marketCap: { ...prev.marketCap, max: Number(e.target.value) }
                  }))}
                  className="terminal-input flex-1 text-xs"
                  placeholder="Max"
                />
              </div>
            </div>

            <div>
              <label className="text-terminal-muted mb-1 block">Daily Volume</label>
              <div className="flex space-x-2">
                <input
                  type="number"
                  value={criteria.volume.min}
                  onChange={(e) => setCriteria(prev => ({
                    ...prev,
                    volume: { ...prev.volume, min: Number(e.target.value) }
                  }))}
                  className="terminal-input flex-1 text-xs"
                  placeholder="Min"
                />
                <input
                  type="number"
                  value={criteria.volume.max}
                  onChange={(e) => setCriteria(prev => ({
                    ...prev,
                    volume: { ...prev.volume, max: Number(e.target.value) }
                  }))}
                  className="terminal-input flex-1 text-xs"
                  placeholder="Max"
                />
              </div>
            </div>
          </div>

          <button
            onClick={runScreener}
            className="mt-3 w-full bg-terminal-orange hover:bg-terminal-amber text-terminal-bg py-2 text-xs font-semibold rounded transition-colors"
          >
            RUN SCREEN
          </button>
        </div>

        {/* Results */}
        <div className="flex-1">
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>Change %</th>
                <th>Volume</th>
                <th>P/E</th>
              </tr>
            </thead>
            <tbody>
              {results.map((stock) => (
                <tr key={stock.symbol}>
                  <td>
                    <div>
                      <div className="font-mono font-semibold text-terminal-cyan text-xs">
                        {stock.symbol}
                      </div>
                      <div className="text-xs text-terminal-muted truncate">
                        {stock.name}
                      </div>
                    </div>
                  </td>
                  <td className="text-right">
                    <div className="financial-number text-xs">
                      ${stock.price.toFixed(2)}
                    </div>
                  </td>
                  <td className="text-right">
                    <div className={`flex items-center justify-end text-xs ${
                      stock.changePercent >= 0 ? 'status-positive' : 'status-negative'
                    }`}>
                      {stock.changePercent >= 0 ? (
                        <TrendingUp className="w-3 h-3 mr-1" />
                      ) : (
                        <TrendingDown className="w-3 h-3 mr-1" />
                      )}
                      <span className="financial-number">
                        {stock.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  </td>
                  <td className="text-right">
                    <div className="financial-number text-xs">
                      {(stock.volume / 1000000).toFixed(1)}M
                    </div>
                  </td>
                  <td className="text-right">
                    <div className="financial-number text-xs">
                      {stock.pe.toFixed(1)}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
