"use client";

import React from 'react';
import { Settings, TrendingUp, TrendingDown } from 'lucide-react';
import { TerminalData } from '@/types';

interface OrderBookProps {
  data: TerminalData;
}

const OrderBook: React.FC<OrderBookProps> = () => {
  const orderBookData = {
    bids: [
      { price: 174.95, size: 1250, orders: 15 },
      { price: 174.90, size: 2100, orders: 23 },
      { price: 174.85, size: 1875, orders: 18 },
      { price: 174.80, size: 3200, orders: 31 },
      { price: 174.75, size: 1600, orders: 12 }
    ],
    asks: [
      { price: 175.05, size: 1150, orders: 12 },
      { price: 175.10, size: 1900, orders: 20 },
      { price: 175.15, size: 2250, orders: 25 },
      { price: 175.20, size: 1700, orders: 16 },
      { price: 175.25, size: 2800, orders: 28 }
    ],
    spread: 0.10,
    lastPrice: 175.00
  };

  const maxSize = Math.max(...orderBookData.bids.map(b => b.size), ...orderBookData.asks.map(a => a.size));

  return (
    <div className="h-full p-6 space-y-6">
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Settings className="w-4 h-4" />
            ORDER BOOK
          </h3>
          <div className="text-xs font-mono">
            SPREAD: ${orderBookData.spread.toFixed(2)}
          </div>
        </div>

        <div className="space-y-4">
          {/* Market Depth Visualization */}
          <div className="grid grid-cols-2 gap-4">
            {/* Bids */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-[var(--gains-color)]" />
                <span className="text-xs font-mono font-bold text-[var(--gains-color)]">BIDS</span>
              </div>
              <div className="space-y-1">
                {orderBookData.bids.map((bid, index) => (
                  <div key={index} className="relative">
                    <div 
                      className="absolute inset-0 bg-green-900/20 rounded"
                      style={{ width: `${(bid.size / maxSize) * 100}%` }}
                    ></div>
                    <div className="relative flex justify-between items-center p-2 text-xs font-mono">
                      <span className="text-[var(--gains-color)] font-bold">${bid.price.toFixed(2)}</span>
                      <span>{bid.size.toLocaleString()}</span>
                      <span className="text-[var(--bloomberg-text-secondary)]">{bid.orders}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Asks */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <TrendingDown className="w-4 h-4 text-[var(--losses-color)]" />
                <span className="text-xs font-mono font-bold text-[var(--losses-color)]">ASKS</span>
              </div>
              <div className="space-y-1">
                {orderBookData.asks.map((ask, index) => (
                  <div key={index} className="relative">
                    <div 
                      className="absolute inset-0 bg-red-900/20 rounded"
                      style={{ width: `${(ask.size / maxSize) * 100}%` }}
                    ></div>
                    <div className="relative flex justify-between items-center p-2 text-xs font-mono">
                      <span className="text-[var(--losses-color)] font-bold">${ask.price.toFixed(2)}</span>
                      <span>{ask.size.toLocaleString()}</span>
                      <span className="text-[var(--bloomberg-text-secondary)]">{ask.orders}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Last Trade */}
          <div className="border-t border-[var(--bloomberg-border)] pt-4">
            <div className="text-center">
              <div className="text-xs text-[var(--bloomberg-text-secondary)] mb-1">LAST TRADE</div>
              <div className="font-mono font-bold text-lg">${orderBookData.lastPrice.toFixed(2)}</div>
            </div>
          </div>

          {/* Order Book Statistics */}
          <div className="grid grid-cols-3 gap-4 border-t border-[var(--bloomberg-border)] pt-4">
            <div className="text-center">
              <div className="text-xs text-[var(--bloomberg-text-secondary)]">BID VOLUME</div>
              <div className="font-mono font-bold text-sm text-[var(--gains-color)]">
                {orderBookData.bids.reduce((sum, bid) => sum + bid.size, 0).toLocaleString()}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--bloomberg-text-secondary)]">ASK VOLUME</div>
              <div className="font-mono font-bold text-sm text-[var(--losses-color)]">
                {orderBookData.asks.reduce((sum, ask) => sum + ask.size, 0).toLocaleString()}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-[var(--bloomberg-text-secondary)]">IMBALANCE</div>
              <div className="font-mono font-bold text-sm">
                {(
                  (orderBookData.bids.reduce((sum, bid) => sum + bid.size, 0) / 
                   orderBookData.asks.reduce((sum, ask) => sum + ask.size, 0)) * 100
                ).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrderBook;