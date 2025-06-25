
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { BarChart, TrendingUp } from 'lucide-react';

interface ChartData {
  time: string;
  price: number;
  volume: number;
}

interface PriceChartProps {
  symbol: string;
}

export const PriceChart = ({ symbol }: PriceChartProps) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [timeframe, setTimeframe] = useState('1D');
  const [currentPrice, setCurrentPrice] = useState(180.25);
  const [priceChange, setPriceChange] = useState(2.45);

  useEffect(() => {
    // Generate mock chart data
    const generateData = () => {
      const data = [];
      const basePrice = 180 + Math.random() * 20;
      
      for (let i = 0; i < 100; i++) {
        const time = new Date(Date.now() - (100 - i) * 60000);
        const price = basePrice + Math.sin(i / 10) * 5 + Math.random() * 2;
        data.push({
          time: time.toLocaleTimeString(),
          price: price,
          volume: Math.floor(Math.random() * 1000000),
        });
      }
      return data;
    };

    setChartData(generateData());
    
    const interval = setInterval(() => {
      setCurrentPrice(prev => prev + (Math.random() - 0.5) * 0.5);
      setPriceChange(prev => prev + (Math.random() - 0.5) * 0.1);
    }, 1000);

    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <div className="terminal-panel h-full flex flex-col">
      {/* Chart Header */}
      <div className="border-b border-terminal-border p-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-mono font-bold text-lg text-terminal-cyan">{symbol}</span>
              <BarChart className="w-4 h-4 text-terminal-muted" />
            </div>
            <div className="flex items-center space-x-4 mt-1">
              <span className="financial-number text-xl font-semibold">
                ${currentPrice.toFixed(2)}
              </span>
              <span className={`flex items-center text-sm ${
                priceChange >= 0 ? 'status-positive' : 'status-negative'
              }`}>
                <TrendingUp className="w-3 h-3 mr-1" />
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({((priceChange / currentPrice) * 100).toFixed(2)}%)
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex space-x-2">
          {['1D', '5D', '1M', '3M', '1Y'].map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                timeframe === tf 
                  ? 'bg-terminal-orange text-terminal-bg' 
                  : 'text-terminal-muted hover:text-terminal-text hover:bg-terminal-border'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis 
              dataKey="time" 
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fill: '#888888' }}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fill: '#888888' }}
              domain={['dataMin - 1', 'dataMax + 1']}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1a1a1a',
                border: '1px solid #333333',
                borderRadius: '4px',
                color: '#ffffff'
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#00d4ff" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#00d4ff' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
