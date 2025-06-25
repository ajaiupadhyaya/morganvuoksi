"use client";

import React from 'react';
import { Target, TrendingUp } from 'lucide-react';
import { TerminalData } from '@/types';

interface OptionsFlowProps {
  data: TerminalData;
}

const OptionsFlow: React.FC<OptionsFlowProps> = () => {
  const optionsData = {
    volume: 1234567,
    putCallRatio: 0.85,
    unusualActivity: [
      { symbol: 'AAPL', type: 'CALL', strike: 175, exp: '2024-02-16', volume: 15000, oi: 45000, premium: 2.85 },
      { symbol: 'TSLA', type: 'PUT', strike: 250, exp: '2024-02-23', volume: 8500, oi: 23000, premium: 8.75 },
      { symbol: 'NVDA', type: 'CALL', strike: 500, exp: '2024-03-15', volume: 12000, oi: 67000, premium: 15.25 }
    ]
  };

  return (
    <div className="h-full p-6 space-y-6">
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Target className="w-4 h-4" />
            OPTIONS FLOW
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <div className="metric-label">TOTAL VOLUME</div>
            <div className="metric-value">{optionsData.volume.toLocaleString()}</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">PUT/CALL RATIO</div>
            <div className="metric-value">{optionsData.putCallRatio.toFixed(2)}</div>
            <div className="metric-change text-[var(--bloomberg-text-secondary)]">
              BEARISH
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">UNUSUAL ACTIVITY</div>
            <div className="metric-value">{optionsData.unusualActivity.length}</div>
            <div className="metric-change text-[var(--bloomberg-warning)]">
              <TrendingUp className="w-3 h-3 inline mr-1" />
              HIGH
            </div>
          </div>
        </div>
      </div>

      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title">UNUSUAL OPTIONS ACTIVITY</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="bloomberg-table">
            <thead>
              <tr>
                <th>SYMBOL</th>
                <th>TYPE</th>
                <th className="text-right">STRIKE</th>
                <th>EXPIRY</th>
                <th className="text-right">VOLUME</th>
                <th className="text-right">OI</th>
                <th className="text-right">PREMIUM</th>
              </tr>
            </thead>
            <tbody>
              {optionsData.unusualActivity.map((option, index) => (
                <tr key={index}>
                  <td className="font-bold">{option.symbol}</td>
                  <td className={option.type === 'CALL' ? 'text-[var(--gains-color)]' : 'text-[var(--losses-color)]'}>
                    {option.type}
                  </td>
                  <td className="text-right">${option.strike}</td>
                  <td>{option.exp}</td>
                  <td className="text-right">{option.volume.toLocaleString()}</td>
                  <td className="text-right">{option.oi.toLocaleString()}</td>
                  <td className="text-right font-semibold">${option.premium.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default OptionsFlow;