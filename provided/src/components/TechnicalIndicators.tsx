
import React, { useState, useEffect } from 'react';
import { ComposedChart, Line, Bar, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, BarChart3, TrendingUp, Layers } from 'lucide-react';

interface IndicatorData {
  time: string;
  price: number;
  volume: number;
  rsi: number;
  macd: number;
  macdSignal: number;
  bb_upper: number;
  bb_middle: number;
  bb_lower: number;
  support: number;
  resistance: number;
}

export const TechnicalIndicators = ({ symbol }: { symbol: string }) => {
  const [data, setData] = useState<IndicatorData[]>([]);
  const [activeIndicators, setActiveIndicators] = useState({
    rsi: true,
    macd: true,
    bollinger: true,
    support_resistance: true
  });

  useEffect(() => {
    const generateTechnicalData = () => {
      const technicalData = [];
      const basePrice = 180;
      
      for (let i = 0; i < 50; i++) {
        const time = new Date(Date.now() - (50 - i) * 300000);
        const price = basePrice + Math.sin(i / 8) * 15 + Math.random() * 5;
        
        // Calculate mock technical indicators
        const rsi = 30 + Math.sin(i / 10) * 40 + Math.random() * 20;
        const macd = Math.sin(i / 12) * 2 + Math.random() * 0.5;
        const macdSignal = macd - 0.2 + Math.random() * 0.4;
        
        // Bollinger Bands
        const bb_middle = price;
        const bb_upper = price + 5 + Math.random() * 2;
        const bb_lower = price - 5 - Math.random() * 2;
        
        // Support and Resistance
        const support = Math.floor(price / 5) * 5 - 2;
        const resistance = Math.ceil(price / 5) * 5 + 2;
        
        technicalData.push({
          time: time.toLocaleTimeString(),
          price,
          volume: Math.floor(Math.random() * 1000000),
          rsi,
          macd,
          macdSignal,
          bb_upper,
          bb_middle,
          bb_lower,
          support,
          resistance
        });
      }
      return technicalData;
    };

    setData(generateTechnicalData());
    const interval = setInterval(() => {
      setData(generateTechnicalData());
    }, 5000);

    return () => clearInterval(interval);
  }, [symbol]);

  const toggleIndicator = (indicator: keyof typeof activeIndicators) => {
    setActiveIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };

  const latestData = data[data.length - 1];

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">TECHNICAL INDICATORS</h2>
          </div>
        </div>
        
        {/* Indicator Controls */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(activeIndicators).map(([key, active]) => (
            <button
              key={key}
              onClick={() => toggleIndicator(key as keyof typeof activeIndicators)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                active 
                  ? 'bg-terminal-orange text-terminal-bg' 
                  : 'bg-terminal-border text-terminal-muted hover:bg-terminal-orange/20'
              }`}
            >
              {key.toUpperCase().replace('_', '/')}
            </button>
          ))}
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        {/* Live Indicator Values */}
        <div className="p-3 border-b border-terminal-border/50">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-terminal-muted">RSI (14)</div>
              <div className={`text-sm font-semibold ${
                latestData?.rsi > 70 ? 'text-terminal-red' : 
                latestData?.rsi < 30 ? 'text-terminal-green' : 'text-terminal-cyan'
              }`}>
                {latestData?.rsi.toFixed(1)}
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">MACD</div>
              <div className={`text-sm font-semibold ${
                (latestData?.macd || 0) > (latestData?.macdSignal || 0) ? 'text-terminal-green' : 'text-terminal-red'
              }`}>
                {latestData?.macd.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
        
        {/* Main Chart */}
        <div className="flex-1 p-3">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data}>
              <XAxis 
                dataKey="time" 
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              <YAxis 
                yAxisId="price"
                orientation="right"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10, fill: '#888888' }}
              />
              
              {/* Price Line */}
              <Line 
                yAxisId="price"
                type="monotone" 
                dataKey="price" 
                stroke="#00d4ff" 
                strokeWidth={2}
                dot={false}
              />
              
              {/* Bollinger Bands */}
              {activeIndicators.bollinger && (
                <>
                  <Line 
                    yAxisId="price"
                    type="monotone" 
                    dataKey="bb_upper" 
                    stroke="#ff6b35" 
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                  <Line 
                    yAxisId="price"
                    type="monotone" 
                    dataKey="bb_lower" 
                    stroke="#ff6b35" 
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                </>
              )}
              
              {/* Support and Resistance */}
              {activeIndicators.support_resistance && (
                <>
                  <ReferenceLine 
                    yAxisId="price"
                    y={latestData?.support} 
                    stroke="#00ff88" 
                    strokeDasharray="5 5"
                    label={{ value: "Support", fontSize: 10, fill: "#00ff88" }}
                  />
                  <ReferenceLine 
                    yAxisId="price"
                    y={latestData?.resistance} 
                    stroke="#ff4757" 
                    strokeDasharray="5 5"
                    label={{ value: "Resistance", fontSize: 10, fill: "#ff4757" }}
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
