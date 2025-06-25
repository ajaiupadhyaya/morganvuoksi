
import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Zap, Target, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';

interface PredictionData {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence: number;
}

interface MarketSignal {
  id: string;
  type: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  reason: string;
  timeframe: string;
  accuracy: number;
}

export const AIMarketAnalysis = ({ symbol }: { symbol: string }) => {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [signals, setSignals] = useState<MarketSignal[]>([]);
  const [modelAccuracy, setModelAccuracy] = useState(0.847);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Generate mock ML predictions
    const generatePredictions = () => {
      const data = [];
      const basePrice = 180;
      
      for (let i = 0; i < 24; i++) {
        const time = new Date(Date.now() + i * 3600000);
        const actual = basePrice + Math.sin(i / 4) * 10 + Math.random() * 5;
        const predicted = actual + (Math.random() - 0.5) * 3;
        const confidence = 0.6 + Math.random() * 0.3;
        
        data.push({
          timestamp: time.toLocaleTimeString(),
          actual: actual,
          predicted: predicted,
          confidence: confidence
        });
      }
      return data;
    };

    const generateSignals = () => {
      return [
        {
          id: '1',
          type: 'bullish' as const,
          strength: 0.85,
          reason: 'LSTM Neural Network detected strong upward momentum pattern',
          timeframe: '4H',
          accuracy: 0.91
        },
        {
          id: '2',
          type: 'bearish' as const,
          strength: 0.62,
          reason: 'Transformer model identifies resistance level convergence',
          timeframe: '1D',
          accuracy: 0.78
        },
        {
          id: '3',
          type: 'bullish' as const,
          strength: 0.73,
          reason: 'Reinforcement Learning agent suggests long position',
          timeframe: '15M',
          accuracy: 0.86
        }
      ];
    };

    setPredictions(generatePredictions());
    setSignals(generateSignals());

    const interval = setInterval(() => {
      setIsProcessing(true);
      setTimeout(() => {
        setPredictions(generatePredictions());
        setModelAccuracy(0.8 + Math.random() * 0.15);
        setIsProcessing(false);
      }, 1500);
    }, 10000);

    return () => clearInterval(interval);
  }, [symbol]);

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'bullish': return 'text-terminal-green';
      case 'bearish': return 'text-terminal-red';
      default: return 'text-terminal-cyan';
    }
  };

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'bullish': return TrendingUp;
      case 'bearish': return TrendingDown;
      default: return Target;
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4 text-terminal-orange" />
          <h2 className="text-sm font-semibold text-terminal-orange">AI MARKET ANALYSIS</h2>
          {isProcessing && <div className="w-2 h-2 bg-terminal-cyan rounded-full animate-pulse" />}
        </div>
        <div className="text-xs text-terminal-muted">
          Model Accuracy: <span className="text-terminal-green">{(modelAccuracy * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {/* ML Predictions Chart */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-2">24H PRICE PREDICTION</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={predictions}>
                <XAxis 
                  dataKey="timestamp" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#888888" 
                  strokeWidth={1}
                  dot={false}
                  name="Actual"
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  name="Predicted"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Signals */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">NEURAL NETWORK SIGNALS</h3>
          <div className="space-y-3">
            {signals.map((signal) => {
              const IconComponent = getSignalIcon(signal.type);
              return (
                <div key={signal.id} className="bg-terminal-bg/50 rounded p-3 border border-terminal-border/30">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <IconComponent className={`w-4 h-4 ${getSignalColor(signal.type)}`} />
                      <span className={`text-sm font-medium ${getSignalColor(signal.type)} uppercase`}>
                        {signal.type}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-terminal-muted">{signal.timeframe}</span>
                      <div className={`w-2 h-2 rounded-full ${
                        signal.strength > 0.8 ? 'bg-terminal-green' : 
                        signal.strength > 0.6 ? 'bg-terminal-amber' : 'bg-terminal-red'
                      }`} />
                    </div>
                  </div>
                  
                  <p className="text-xs text-terminal-text mb-2">{signal.reason}</p>
                  
                  <div className="flex justify-between items-center">
                    <div className="text-xs text-terminal-muted">
                      Strength: <span className="text-terminal-cyan">{(signal.strength * 100).toFixed(0)}%</span>
                    </div>
                    <div className="text-xs text-terminal-muted">
                      Accuracy: <span className="text-terminal-green">{(signal.accuracy * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
