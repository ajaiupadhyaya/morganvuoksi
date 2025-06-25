"use client";

import React from 'react';
import { Brain, TrendingUp, Target, Zap } from 'lucide-react';
import { TerminalData } from '@/types';

interface AIMarketAnalysisProps {
  data: TerminalData;
}

const AIMarketAnalysis: React.FC<AIMarketAnalysisProps> = () => {
  const aiData = {
    sentiment: 0.72,
    prediction: 'BULLISH',
    confidence: 85,
    targetPrice: 185.50,
    signals: [
      { type: 'BUY', strength: 8.5, reason: 'Technical breakout pattern detected' },
      { type: 'HOLD', strength: 6.2, reason: 'Earnings momentum positive' },
      { type: 'SELL', strength: 3.1, reason: 'Overbought RSI levels' }
    ]
  };

  return (
    <div className="h-full p-6 space-y-6">
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Brain className="w-4 h-4" />
            AI MARKET ANALYSIS
          </h3>
          <div className="flex items-center gap-2">
            <span className="status-indicator status-live"></span>
            <span className="text-xs font-mono">ML MODELS</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="metric-card">
            <div className="metric-label">AI SENTIMENT</div>
            <div className="metric-value text-[var(--bloomberg-terminal-green)]">
              {(aiData.sentiment * 100).toFixed(0)}%
            </div>
            <div className="metric-change positive-change">
              <TrendingUp className="w-3 h-3 inline mr-1" />
              BULLISH
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">PREDICTION</div>
            <div className="metric-value text-[var(--gains-color)]">{aiData.prediction}</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">CONFIDENCE</div>
            <div className="metric-value">{aiData.confidence}%</div>
            <div className="metric-change text-[var(--bloomberg-terminal-green)]">
              <Zap className="w-3 h-3 inline mr-1" />
              HIGH
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-label">TARGET PRICE</div>
            <div className="metric-value">${aiData.targetPrice.toFixed(2)}</div>
            <div className="metric-change positive-change">
              <Target className="w-3 h-3 inline mr-1" />
              +5.8%
            </div>
          </div>
        </div>
      </div>

      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title">AI TRADING SIGNALS</h3>
        </div>

        <div className="space-y-4">
          {aiData.signals.map((signal, index) => (
            <div key={index} className="p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 rounded text-xs font-bold ${
                    signal.type === 'BUY' ? 'bg-green-900/30 text-[var(--gains-color)]' :
                    signal.type === 'SELL' ? 'bg-red-900/30 text-[var(--losses-color)]' :
                    'bg-gray-900/30 text-[var(--neutral-color)]'
                  }`}>
                    {signal.type}
                  </span>
                  <span className="font-mono text-sm">Strength: {signal.strength}/10</span>
                </div>
                <div className="w-24 bg-[var(--bloomberg-tertiary)] rounded-full h-2 overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      signal.strength > 7 ? 'bg-[var(--gains-color)]' :
                      signal.strength > 4 ? 'bg-[var(--bloomberg-warning)]' :
                      'bg-[var(--losses-color)]'
                    }`}
                    style={{ width: `${(signal.strength / 10) * 100}%` }}
                  ></div>
                </div>
              </div>
              <div className="text-sm text-[var(--bloomberg-text-secondary)]">{signal.reason}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AIMarketAnalysis;