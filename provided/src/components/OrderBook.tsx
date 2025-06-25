
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface OrderBookEntry {
  price: number;
  size: number;
  total: number;
}

interface OrderBookProps {
  symbol: string;
}

export const OrderBook = ({ symbol }: OrderBookProps) => {
  const [bids, setBids] = useState<OrderBookEntry[]>([]);
  const [asks, setAsks] = useState<OrderBookEntry[]>([]);
  const [spread, setSpread] = useState(0.02);

  useEffect(() => {
    // Generate mock order book data
    const generateOrders = () => {
      const basePrice = 180.25;
      const newBids: OrderBookEntry[] = [];
      const newAsks: OrderBookEntry[] = [];
      
      let bidTotal = 0;
      let askTotal = 0;
      
      // Generate bids (below market price)
      for (let i = 0; i < 8; i++) {
        const price = basePrice - (i + 1) * 0.01;
        const size = Math.floor(Math.random() * 500) + 100;
        bidTotal += size;
        newBids.push({ price, size, total: bidTotal });
      }
      
      // Generate asks (above market price)
      for (let i = 0; i < 8; i++) {
        const price = basePrice + (i + 1) * 0.01;
        const size = Math.floor(Math.random() * 500) + 100;
        askTotal += size;
        newAsks.push({ price, size, total: askTotal });
      }
      
      setBids(newBids);
      setAsks(newAsks.reverse());
      setSpread(newAsks[0]?.price - newBids[0]?.price || 0.02);
    };

    generateOrders();
    const interval = setInterval(generateOrders, 2000);
    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-terminal-orange">ORDER BOOK</h2>
        <div className="text-xs text-terminal-muted">
          Spread: <span className="text-terminal-cyan">${spread.toFixed(2)}</span>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <div className="h-full flex flex-col">
          {/* Header */}
          <div className="grid grid-cols-3 gap-2 px-3 py-2 text-xs font-medium text-terminal-muted border-b border-terminal-border/50">
            <div className="text-left">Price</div>
            <div className="text-right">Size</div>
            <div className="text-right">Total</div>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {/* Asks (Sell orders) */}
            <div className="space-y-0.5">
              {asks.map((ask, index) => (
                <div key={`ask-${index}`} className="grid grid-cols-3 gap-2 px-3 py-1 text-xs hover:bg-terminal-panel/50">
                  <div className="text-terminal-red financial-number">{ask.price.toFixed(2)}</div>
                  <div className="text-right financial-number">{ask.size.toLocaleString()}</div>
                  <div className="text-right text-terminal-muted financial-number">{ask.total.toLocaleString()}</div>
                </div>
              ))}
            </div>
            
            {/* Spread indicator */}
            <div className="flex items-center justify-center py-2 border-y border-terminal-border/30 my-1">
              <div className="flex items-center space-x-2 text-xs">
                <TrendingDown className="w-3 h-3 text-terminal-red" />
                <span className="text-terminal-cyan font-medium">${spread.toFixed(2)}</span>
                <TrendingUp className="w-3 h-3 text-terminal-green" />
              </div>
            </div>
            
            {/* Bids (Buy orders) */}
            <div className="space-y-0.5">
              {bids.map((bid, index) => (
                <div key={`bid-${index}`} className="grid grid-cols-3 gap-2 px-3 py-1 text-xs hover:bg-terminal-panel/50">
                  <div className="text-terminal-green financial-number">{bid.price.toFixed(2)}</div>
                  <div className="text-right financial-number">{bid.size.toLocaleString()}</div>
                  <div className="text-right text-terminal-muted financial-number">{bid.total.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
