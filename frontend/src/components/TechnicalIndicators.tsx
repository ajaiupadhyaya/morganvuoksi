"use client";

import React from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';
import { TerminalData } from '@/types';

interface TechnicalIndicatorsProps {
  data: TerminalData;
}

const TechnicalIndicators: React.FC<TechnicalIndicatorsProps> = () => {
  const indicators = {
    rsi: 68.5,
    macd: { value: 2.45, signal: 1.87, histogram: 0.58 },
    bb: { upper: 178.50, middle: 175.25, lower: 172.00 },
    sma20: 174.50,
    sma50: 170.25,
    volume: 1.2
  };

  const getRSIColor = (rsi: number) => {
    if (rsi > 70) return 'text-[var(--losses-color)]';
    if (rsi < 30) return 'text-[var(--gains-color)]';
    return 'text-[var(--neutral-color)]';
  };

  return (
    <div className="bloomberg-card h-full">
      <div className="bloomberg-card-header">
        <h3 className="bloomberg-card-title flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          TECHNICAL INDICATORS
        </h3>
      </div>

      <div className="space-y-4">
        {/* RSI */}
        <div className="p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-bold">RSI (14)</span>
            <span className={`font-mono font-bold ${getRSIColor(indicators.rsi)}`}>
              {indicators.rsi.toFixed(1)}
            </span>
          </div>
          <div className="w-full bg-[var(--bloomberg-tertiary)] rounded-full h-2 overflow-hidden">
            <div 
              className="h-full bg-[var(--bloomberg-blue)] transition-all duration-500"
              style={{ width: `${indicators.rsi}%` }}
            ></div>
          </div>
          <div className="flex justify-between mt-1 text-xs text-[var(--bloomberg-text-secondary)]">
            <span>Oversold (30)</span>
            <span>Overbought (70)</span>
          </div>
        </div>

        {/* MACD */}
        <div className="p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
          <div className="text-xs font-mono font-bold mb-2">MACD</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-xs">MACD Line</span>
              <span className="font-mono text-xs">{indicators.macd.value.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs">Signal Line</span>
              <span className="font-mono text-xs">{indicators.macd.signal.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs">Histogram</span>
              <span className={`font-mono text-xs ${indicators.macd.histogram > 0 ? 'text-[var(--gains-color)]' : 'text-[var(--losses-color)]'}`}>
                {indicators.macd.histogram > 0 ? '+' : ''}{indicators.macd.histogram.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Bollinger Bands */}
        <div className="p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
          <div className="text-xs font-mono font-bold mb-2">BOLLINGER BANDS</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-xs">Upper</span>
              <span className="font-mono text-xs">${indicators.bb.upper.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs">Middle (SMA20)</span>
              <span className="font-mono text-xs">${indicators.bb.middle.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-xs">Lower</span>
              <span className="font-mono text-xs">${indicators.bb.lower.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Moving Averages */}
        <div className="p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
          <div className="text-xs font-mono font-bold mb-2">MOVING AVERAGES</div>
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs">SMA 20</span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs">${indicators.sma20.toFixed(2)}</span>
                <TrendingUp className="w-3 h-3 text-[var(--gains-color)]" />
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs">SMA 50</span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs">${indicators.sma50.toFixed(2)}</span>
                <TrendingUp className="w-3 h-3 text-[var(--gains-color)]" />
              </div>
            </div>
          </div>
        </div>

        {/* Volume Analysis */}
        <div className="p-3 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono font-bold">VOLUME</span>
            <span className="font-mono text-xs">{indicators.volume.toFixed(1)}x</span>
          </div>
          <div className="text-xs text-[var(--bloomberg-text-secondary)]">
            vs 20-day average
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalIndicators;