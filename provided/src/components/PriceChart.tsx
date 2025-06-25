import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, AreaChart, Area, ReferenceLine } from 'recharts';
import { BarChart, TrendingUp, TrendingDown, Activity, Target, Zap, Clock, Volume2, AlertCircle } from 'lucide-react';

interface ChartData {
  time: string;
  timestamp: number;
  price: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
  sma20: number;
  sma50: number;
  rsi: number;
  macd: number;
}

interface PriceChartProps {
  symbol: string;
}

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'bullish' | 'bearish' | 'neutral';
  description: string;
}

export const PriceChart = ({ symbol }: PriceChartProps) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState<'line' | 'area' | 'candlestick'>('line');
  const [currentPrice, setCurrentPrice] = useState(180.25);
  const [priceChange, setPriceChange] = useState(2.45);
  const [dayHigh, setDayHigh] = useState(182.50);
  const [dayLow, setDayLow] = useState(177.80);
  const [volume, setVolume] = useState(85420000);
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [showTechnicals, setShowTechnicals] = useState(true);
  const [isLive, setIsLive] = useState(true);

  useEffect(() => {
    // Generate sophisticated mock chart data
    const generateData = () => {
      const data: ChartData[] = [];
      const basePrice = 180 + Math.random() * 20;
      const points = timeframe === '1D' ? 390 : timeframe === '5D' ? 1950 : 7800; // Market minutes
      
      for (let i = 0; i < points; i++) {
        const timestamp = Date.now() - (points - i) * 60000;
        const time = new Date(timestamp);
        
        // More realistic price movement with volatility clustering
        const volatility = 0.5 + Math.sin(i / 50) * 0.3;
        const trend = Math.sin(i / 100) * 2;
        const noise = (Math.random() - 0.5) * volatility;
        const price = Math.max(0.01, basePrice + trend + noise + Math.sin(i / 20) * 1.5);
        
        // Calculate technical indicators
        const sma20 = i >= 20 ? data.slice(i-20, i).reduce((sum, d) => sum + d.price, 0) / 20 : price;
        const sma50 = i >= 50 ? data.slice(i-50, i).reduce((sum, d) => sum + d.price, 0) / 50 : price;
        
        data.push({
          time: timeframe === '1D' ? time.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false 
          }) : time.toLocaleDateString(),
          timestamp,
          price: price,
          volume: Math.floor(Math.random() * 2000000 + 500000),
          high: price + Math.random() * 0.5,
          low: price - Math.random() * 0.5,
          open: i > 0 ? data[i-1].close : price,
          close: price,
          sma20: sma20,
          sma50: sma50,
          rsi: 30 + Math.random() * 40, // RSI between 30-70
          macd: (Math.random() - 0.5) * 2,
        });
      }
      return data;
    };

    setChartData(generateData());
    
    // Update technical indicators
    const newIndicators: TechnicalIndicator[] = [
      {
        name: 'RSI (14)',
        value: 58.32,
        signal: 'neutral',
        description: 'Relative Strength Index - Momentum oscillator'
      },
      {
        name: 'MACD',
        value: 1.25,
        signal: 'bullish',
        description: 'Moving Average Convergence Divergence'
      },
      {
        name: 'BOLLINGER',
        value: 0.85,
        signal: 'neutral',
        description: 'Bollinger Bands - Volatility indicator'
      },
      {
        name: 'VOLUME',
        value: 142.3,
        signal: 'bullish',
        description: 'Volume relative to 20-day average'
      }
    ];
    
    setIndicators(newIndicators);
    
    const interval = setInterval(() => {
      if (isLive) {
        setCurrentPrice(prev => {
          const newPrice = prev + (Math.random() - 0.5) * 0.5;
          return Math.max(0.01, newPrice);
        });
        setPriceChange(prev => prev + (Math.random() - 0.5) * 0.1);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [symbol, timeframe, isLive]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish': return 'status-positive';
      case 'bearish': return 'status-negative';
      default: return 'status-neutral';
    }
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="terminal-panel p-3 border-2 border-terminal-cyan bg-terminal-bg">
          <div className="text-terminal-cyan font-mono text-xs mb-2">{label}</div>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="text-xs font-mono">
              <span className="text-terminal-muted">{entry.name}: </span>
              <span className="text-terminal-text font-bold">
                ${entry.value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="terminal-panel h-full flex flex-col bg-terminal-bg border-2 border-terminal-border">
      {/* Professional Chart Header */}
      <div className="border-b-2 border-terminal-orange p-3 bg-gradient-to-r from-terminal-panel to-terminal-bg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-terminal-orange terminal-pulse" />
              <span className="font-mono font-bold text-lg text-terminal-cyan tracking-wider">{symbol}</span>
              <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-terminal-green' : 'bg-terminal-red'} terminal-pulse`} />
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex flex-col">
                <span className="financial-number text-2xl font-bold text-terminal-text">
                  ${currentPrice.toFixed(2)}
                </span>
                <div className={`flex items-center space-x-2 ${
                  priceChange >= 0 ? 'status-positive' : 'status-negative'
                }`}>
                  {priceChange >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span className="financial-number font-bold">
                    {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} 
                    ({((priceChange / currentPrice) * 100).toFixed(2)}%)
                  </span>
                </div>
              </div>
              
              <div className="flex flex-col text-xs space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="text-terminal-muted font-mono">HIGH:</span>
                  <span className="text-terminal-green font-mono font-bold">${dayHigh.toFixed(2)}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-terminal-muted font-mono">LOW:</span>
                  <span className="text-terminal-red font-mono font-bold">${dayLow.toFixed(2)}</span>
                </div>
              </div>
              
              <div className="flex flex-col text-xs space-y-1">
                <div className="flex items-center space-x-2">
                  <Volume2 className="w-3 h-3 text-terminal-muted" />
                  <span className="text-terminal-muted font-mono">VOLUME:</span>
                  <span className="text-terminal-cyan font-mono font-bold">{formatVolume(volume)}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="w-3 h-3 text-terminal-muted" />
                  <span className="text-terminal-muted font-mono">LAST:</span>
                  <span className="text-terminal-text font-mono">{new Date().toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsLive(!isLive)}
              className={`terminal-button flex items-center space-x-1 ${
                isLive ? 'border-terminal-green text-terminal-green' : 'border-terminal-red text-terminal-red'
              }`}
            >
              <Zap className="w-3 h-3" />
              <span>{isLive ? 'LIVE' : 'PAUSED'}</span>
            </button>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          {/* Timeframe Selection */}
          <div className="flex space-x-1">
            {['1D', '5D', '1M', '3M', '6M', '1Y', '5Y'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-1 text-xs font-mono font-bold uppercase transition-all duration-150 border ${
                  timeframe === tf 
                    ? 'bg-terminal-orange text-terminal-bg border-terminal-orange glow-orange' 
                    : 'text-terminal-muted hover:text-terminal-text hover:border-terminal-cyan border-terminal-border'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          
          {/* Chart Type Selection */}
          <div className="flex space-x-1">
            {[
              { type: 'line', label: 'LINE', icon: Activity },
              { type: 'area', label: 'AREA', icon: BarChart },
              { type: 'candlestick', label: 'CANDLE', icon: Target }
            ].map(({ type, label, icon: Icon }) => (
              <button
                key={type}
                onClick={() => setChartType(type as any)}
                className={`px-2 py-1 text-xs font-mono font-bold uppercase transition-all duration-150 border flex items-center space-x-1 ${
                  chartType === type 
                    ? 'bg-terminal-cyan text-terminal-bg border-terminal-cyan glow-cyan' 
                    : 'text-terminal-muted hover:text-terminal-text hover:border-terminal-cyan border-terminal-border'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{label}</span>
              </button>
            ))}
          </div>
          
          {/* Technical Analysis Toggle */}
          <button
            onClick={() => setShowTechnicals(!showTechnicals)}
            className={`terminal-button flex items-center space-x-1 ${
              showTechnicals ? 'border-terminal-green text-terminal-green' : 'border-terminal-border text-terminal-muted'
            }`}
          >
            <AlertCircle className="w-3 h-3" />
            <span>TECHNICALS</span>
          </button>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Main Chart Area */}
        <div className="flex-1 p-2 terminal-chart">
          <ResponsiveContainer width="100%" height="100%">
            {chartType === 'area' ? (
              <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888', fontFamily: 'monospace' }}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888', fontFamily: 'monospace' }}
                  domain={['dataMin - 1', 'dataMax + 1']}
                  orientation="right"
                />
                <Tooltip content={<CustomTooltip />} />
                {showTechnicals && (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="sma20" 
                      stroke="#ffa500" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="3 3"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="sma50" 
                      stroke="#ff6b35" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="5 5"
                    />
                  </>
                )}
                <Area 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  fill="url(#priceGradient)"
                />
                <ReferenceLine y={currentPrice} stroke="#ffffff" strokeDasharray="2 2" strokeWidth={1} />
              </AreaChart>
            ) : (
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <XAxis 
                  dataKey="time" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888', fontFamily: 'monospace' }}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: '#888888', fontFamily: 'monospace' }}
                  domain={['dataMin - 1', 'dataMax + 1']}
                  orientation="right"
                />
                <Tooltip content={<CustomTooltip />} />
                {showTechnicals && (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="sma20" 
                      stroke="#ffa500" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="3 3"
                      name="SMA 20"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="sma50" 
                      stroke="#ff6b35" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="5 5"
                      name="SMA 50"
                    />
                  </>
                )}
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#00d4ff" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#00d4ff', stroke: '#ffffff', strokeWidth: 2 }}
                  name="Price"
                />
                <ReferenceLine y={currentPrice} stroke="#ffffff" strokeDasharray="2 2" strokeWidth={1} />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>

        {/* Technical Indicators Panel */}
        {showTechnicals && (
          <div className="w-64 border-l-2 border-terminal-border bg-terminal-panel/50 p-2">
            <div className="mb-3">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-4 h-4 text-terminal-orange" />
                <span className="text-terminal-orange font-mono font-bold text-sm uppercase">
                  TECHNICAL ANALYSIS
                </span>
              </div>
            </div>
            
            <div className="space-y-3">
              {indicators.map((indicator, index) => (
                <div key={index} className="terminal-panel p-2 border border-terminal-border/50">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-terminal-cyan font-mono font-bold text-xs">
                      {indicator.name}
                    </span>
                    <span className={`text-xs font-mono font-bold ${getSignalColor(indicator.signal)}`}>
                      {indicator.signal.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-terminal-text font-mono font-bold">
                    {indicator.value.toFixed(2)}
                  </div>
                  <div className="text-terminal-muted text-xs font-mono mt-1">
                    {indicator.description}
                  </div>
                </div>
              ))}
            </div>
            
            {/* Quick Analysis Summary */}
            <div className="mt-4 terminal-panel p-2 border-2 border-terminal-orange">
              <div className="text-terminal-orange font-mono font-bold text-xs mb-2 uppercase">
                AI SIGNAL STRENGTH
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-terminal-bg h-2 rounded">
                  <div 
                    className="h-2 bg-gradient-to-r from-terminal-green to-terminal-orange rounded"
                    style={{ width: '68%' }}
                  />
                </div>
                <span className="text-terminal-green font-mono font-bold text-sm">68%</span>
              </div>
              <div className="text-terminal-muted text-xs font-mono mt-1">
                BULLISH MOMENTUM DETECTED
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
