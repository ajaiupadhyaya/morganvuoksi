"use client";
import { FC } from "react";
import { Activity, TrendingUp, TrendingDown, Clock } from "lucide-react";

interface Trade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  status: 'FILLED' | 'PENDING' | 'PARTIAL';
  pnl?: number;
}

// Enhanced mock trading data
const trades: Trade[] = [
  { 
    id: 'TXN001', 
    symbol: 'AAPL', 
    type: 'BUY', 
    quantity: 100, 
    price: 175.25, 
    timestamp: '09:30:15',
    status: 'FILLED',
    pnl: 245.50
  },
  { 
    id: 'TXN002', 
    symbol: 'GOOGL', 
    type: 'SELL', 
    quantity: 25, 
    price: 2687.34, 
    timestamp: '09:32:45',
    status: 'FILLED',
    pnl: -123.75
  },
  { 
    id: 'TXN003', 
    symbol: 'TSLA', 
    type: 'BUY', 
    quantity: 50, 
    price: 267.89, 
    timestamp: '09:35:22',
    status: 'PARTIAL',
    pnl: 89.25
  },
  { 
    id: 'TXN004', 
    symbol: 'MSFT', 
    type: 'SELL', 
    quantity: 75, 
    price: 323.45, 
    timestamp: '09:38:10',
    status: 'PENDING'
  },
  { 
    id: 'TXN005', 
    symbol: 'NVDA', 
    type: 'BUY', 
    quantity: 30, 
    price: 434.56, 
    timestamp: '09:40:33',
    status: 'FILLED',
    pnl: 567.89
  },
];

const DataTable: FC = () => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'FILLED':
        return 'text-[var(--bloomberg-terminal-green)]';
      case 'PENDING':
        return 'text-[var(--bloomberg-warning)]';
      case 'PARTIAL':
        return 'text-[var(--bloomberg-blue)]';
      default:
        return 'text-[var(--bloomberg-text-secondary)]';
    }
  };

  const getPnLColor = (pnl?: number) => {
    if (!pnl) return 'text-[var(--bloomberg-text-secondary)]';
    return pnl >= 0 ? 'text-[var(--gains-color)]' : 'text-[var(--losses-color)]';
  };

  return (
    <div className="bloomberg-card h-full">
      <div className="bloomberg-card-header">
        <h3 className="bloomberg-card-title flex items-center gap-2">
          <Activity className="w-4 h-4" />
          TRADING ACTIVITY
        </h3>
        <div className="flex items-center gap-2">
          <span className="status-indicator status-live"></span>
          <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
            {trades.length} TRADES
          </span>
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="bloomberg-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>SYMBOL</th>
              <th>TYPE</th>
              <th className="text-right">QTY</th>
              <th className="text-right">PRICE</th>
              <th className="text-center">STATUS</th>
              <th className="text-right">P&L</th>
              <th>TIME</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade, index) => (
              <tr key={index}>
                <td className="font-mono text-xs text-[var(--bloomberg-blue)]">
                  {trade.id}
                </td>
                <td className="font-bold text-[var(--bloomberg-text-primary)]">
                  {trade.symbol}
                </td>
                <td className={`font-mono text-xs ${
                  trade.type === 'BUY' 
                    ? 'text-[var(--gains-color)]' 
                    : 'text-[var(--losses-color)]'
                }`}>
                  <div className="flex items-center gap-1">
                    {trade.type === 'BUY' ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    {trade.type}
                  </div>
                </td>
                <td className="text-right font-mono">
                  {trade.quantity.toLocaleString()}
                </td>
                <td className="text-right font-mono font-semibold">
                  ${trade.price.toFixed(2)}
                </td>
                <td className="text-center">
                  <span className={`px-2 py-1 rounded text-xs font-mono font-bold ${getStatusColor(trade.status)}`}>
                    {trade.status}
                  </span>
                </td>
                <td className={`text-right font-mono font-bold ${getPnLColor(trade.pnl)}`}>
                  {trade.pnl ? (
                    <>
                      {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="font-mono text-xs text-[var(--bloomberg-text-secondary)]">
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {trade.timestamp}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Statistics */}
      <div className="mt-4 pt-4 border-t border-[var(--bloomberg-border)]">
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-xs text-[var(--bloomberg-text-secondary)]">TOTAL TRADES</div>
            <div className="font-mono font-bold text-sm">{trades.length}</div>
          </div>
          <div>
            <div className="text-xs text-[var(--bloomberg-text-secondary)]">FILLED</div>
            <div className="font-mono font-bold text-sm text-[var(--bloomberg-terminal-green)]">
              {trades.filter(t => t.status === 'FILLED').length}
            </div>
          </div>
          <div>
            <div className="text-xs text-[var(--bloomberg-text-secondary)]">PENDING</div>
            <div className="font-mono font-bold text-sm text-[var(--bloomberg-warning)]">
              {trades.filter(t => t.status === 'PENDING').length}
            </div>
          </div>
          <div>
            <div className="text-xs text-[var(--bloomberg-text-secondary)]">TOTAL P&L</div>
            <div className={`font-mono font-bold text-sm ${
              trades.reduce((sum, t) => sum + (t.pnl || 0), 0) >= 0 
                ? 'text-[var(--gains-color)]' 
                : 'text-[var(--losses-color)]'
            }`}>
              {trades.reduce((sum, t) => sum + (t.pnl || 0), 0) >= 0 ? '+' : ''}
              ${trades.reduce((sum, t) => sum + (t.pnl || 0), 0).toFixed(2)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataTable; 