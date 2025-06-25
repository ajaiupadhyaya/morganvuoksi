
import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Zap, Shield, Target, DollarSign } from 'lucide-react';

interface TradingInterfaceProps {
  symbol: string;
}

export const TradingInterface = ({ symbol }: TradingInterfaceProps) => {
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('limit');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState('100');
  const [price, setPrice] = useState('180.25');
  const [leverage, setLeverage] = useState(1);

  const handleSubmitOrder = () => {
    console.log('Order submitted:', { symbol, orderType, side, quantity, price, leverage });
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center space-x-2 mb-2">
          <Zap className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">TRADING</h2>
        </div>
        <div className="text-xs text-terminal-muted">{symbol} • $180.25</div>
      </div>
      
      <div className="flex-1 p-3 space-y-4">
        {/* Order Type Selection */}
        <div>
          <div className="text-xs text-terminal-muted mb-2">ORDER TYPE</div>
          <div className="grid grid-cols-3 gap-1">
            {['market', 'limit', 'stop'].map((type) => (
              <button
                key={type}
                onClick={() => setOrderType(type as any)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  orderType === type 
                    ? 'bg-terminal-orange text-terminal-bg' 
                    : 'bg-terminal-border text-terminal-muted hover:bg-terminal-orange/20'
                }`}
              >
                {type.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Buy/Sell Toggle */}
        <div>
          <div className="grid grid-cols-2 gap-1">
            <button
              onClick={() => setSide('buy')}
              className={`flex items-center justify-center space-x-2 py-2 text-sm font-semibold rounded transition-colors ${
                side === 'buy' 
                  ? 'bg-terminal-green/20 text-terminal-green border border-terminal-green' 
                  : 'bg-terminal-border text-terminal-muted hover:bg-terminal-green/10'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>BUY</span>
            </button>
            <button
              onClick={() => setSide('sell')}
              className={`flex items-center justify-center space-x-2 py-2 text-sm font-semibold rounded transition-colors ${
                side === 'sell' 
                  ? 'bg-terminal-red/20 text-terminal-red border border-terminal-red' 
                  : 'bg-terminal-border text-terminal-muted hover:bg-terminal-red/10'
              }`}
            >
              <TrendingDown className="w-4 h-4" />
              <span>SELL</span>
            </button>
          </div>
        </div>

        {/* Quantity Input */}
        <div>
          <div className="text-xs text-terminal-muted mb-1">QUANTITY</div>
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            className="terminal-input w-full"
            placeholder="Shares"
          />
        </div>

        {/* Price Input */}
        {orderType !== 'market' && (
          <div>
            <div className="text-xs text-terminal-muted mb-1">PRICE</div>
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              className="terminal-input w-full"
              placeholder="0.00"
              step="0.01"
            />
          </div>
        )}

        {/* Leverage Selector */}
        <div>
          <div className="text-xs text-terminal-muted mb-2">LEVERAGE</div>
          <div className="flex space-x-1">
            {[1, 2, 5, 10].map((lev) => (
              <button
                key={lev}
                onClick={() => setLeverage(lev)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  leverage === lev 
                    ? 'bg-terminal-cyan text-terminal-bg' 
                    : 'bg-terminal-border text-terminal-muted hover:bg-terminal-cyan/20'
                }`}
              >
                {lev}x
              </button>
            ))}
          </div>
        </div>

        {/* Order Summary */}
        <div className="bg-terminal-bg p-2 rounded border border-terminal-border/50">
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-terminal-muted">Est. Cost:</span>
              <span className="text-terminal-text financial-number">
                ${(parseFloat(quantity) * parseFloat(price)).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-terminal-muted">Buying Power:</span>
              <span className="text-terminal-green financial-number">$125,000</span>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmitOrder}
          className={`w-full py-2 text-sm font-semibold rounded transition-all duration-200 hover:scale-105 ${
            side === 'buy' 
              ? 'bg-terminal-green hover:bg-terminal-green/80 text-terminal-bg glow-green' 
              : 'bg-terminal-red hover:bg-terminal-red/80 text-terminal-bg glow-red'
          }`}
        >
          {side === 'buy' ? 'PLACE BUY ORDER' : 'PLACE SELL ORDER'}
        </button>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-2">
          <button className="flex items-center justify-center space-x-1 py-1 text-xs bg-terminal-border hover:bg-terminal-orange/20 rounded transition-colors">
            <Shield className="w-3 h-3" />
            <span>STOP LOSS</span>
          </button>
          <button className="flex items-center justify-center space-x-1 py-1 text-xs bg-terminal-border hover:bg-terminal-cyan/20 rounded transition-colors">
            <Target className="w-3 h-3" />
            <span>TAKE PROFIT</span>
          </button>
        </div>
      </div>
    </div>
  );
};
