
import React, { useState, useEffect, useRef } from 'react';
import { Search, Command, Clock } from 'lucide-react';

interface CommandPaletteProps {
  onClose: () => void;
  onSymbolSelect: (symbol: string) => void;
}

interface SearchResult {
  symbol: string;
  name: string;
  type: 'stock' | 'etf' | 'index' | 'crypto';
  exchange: string;
}

export const CommandPalette = ({ onClose, onSymbolSelect }: CommandPaletteProps) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const mockResults: SearchResult[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', type: 'stock', exchange: 'NASDAQ' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', type: 'stock', exchange: 'NASDAQ' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', type: 'stock', exchange: 'NASDAQ' },
    { symbol: 'TSLA', name: 'Tesla Inc.', type: 'stock', exchange: 'NASDAQ' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', type: 'stock', exchange: 'NASDAQ' },
    { symbol: 'SPY', name: 'SPDR S&P 500 ETF Trust', type: 'etf', exchange: 'NYSE' },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust', type: 'etf', exchange: 'NASDAQ' },
    { symbol: 'BTC-USD', name: 'Bitcoin USD', type: 'crypto', exchange: 'Crypto' },
  ];

  useEffect(() => {
    if (query.length > 0) {
      const filtered = mockResults.filter(
        result =>
          result.symbol.toLowerCase().includes(query.toLowerCase()) ||
          result.name.toLowerCase().includes(query.toLowerCase())
      );
      setResults(filtered);
      setSelectedIndex(0);
    } else {
      setResults([]);
    }
  }, [query]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev + 1) % results.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev - 1 + results.length) % results.length);
    } else if (e.key === 'Enter' && results[selectedIndex]) {
      handleSelect(results[selectedIndex]);
    }
  };

  const handleSelect = (result: SearchResult) => {
    onSymbolSelect(result.symbol);
    onClose();
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'stock': return 'text-terminal-cyan';
      case 'etf': return 'text-terminal-green';
      case 'index': return 'text-terminal-amber';
      case 'crypto': return 'text-terminal-orange';
      default: return 'text-terminal-muted';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-start justify-center pt-20 z-50">
      <div className="bg-terminal-panel border border-terminal-border rounded-lg w-full max-w-2xl mx-4 shadow-2xl">
        {/* Search Input */}
        <div className="flex items-center border-b border-terminal-border p-4">
          <Search className="w-5 h-5 text-terminal-muted mr-3" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search symbols, companies, or commands..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent text-terminal-text placeholder-terminal-muted focus:outline-none"
          />
          <div className="flex items-center space-x-2 text-xs text-terminal-muted">
            <kbd className="bg-terminal-border px-2 py-1 rounded">ESC</kbd>
            <span>to close</span>
          </div>
        </div>

        {/* Results */}
        <div className="max-h-96 overflow-y-auto">
          {query.length === 0 ? (
            <div className="p-6 text-center">
              <Command className="w-8 h-8 text-terminal-muted mx-auto mb-2" />
              <p className="text-terminal-muted text-sm">
                Start typing to search for symbols, companies, or use commands
              </p>
              <div className="mt-4 space-y-2 text-xs text-terminal-muted">
                <div>Try: AAPL, GOOGL, SPY, or BTC-USD</div>
              </div>
            </div>
          ) : results.length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-terminal-muted text-sm">
                No results found for "{query}"
              </p>
            </div>
          ) : (
            <div className="py-2">
              {results.map((result, index) => (
                <div
                  key={result.symbol}
                  className={`flex items-center justify-between px-4 py-3 cursor-pointer transition-colors ${
                    index === selectedIndex
                      ? 'bg-terminal-orange/20 border-l-2 border-terminal-orange'
                      : 'hover:bg-terminal-border/30'
                  }`}
                  onClick={() => handleSelect(result)}
                >
                  <div className="flex items-center space-x-3">
                    <div>
                      <div className="font-mono font-semibold text-terminal-cyan">
                        {result.symbol}
                      </div>
                      <div className="text-sm text-terminal-muted truncate">
                        {result.name}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <span className={`text-xs font-medium px-2 py-1 rounded ${getTypeColor(result.type)}`}>
                      {result.type.toUpperCase()}
                    </span>
                    <span className="text-xs text-terminal-muted">
                      {result.exchange}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
