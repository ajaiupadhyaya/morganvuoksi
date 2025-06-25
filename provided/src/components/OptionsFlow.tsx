
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { TrendingUp, TrendingDown, Circle, Target } from 'lucide-react';

interface OptionsFlowData {
  strike: number;
  calls: number;
  puts: number;
  callVolume: number;
  putVolume: number;
  oi: number;
  type: 'calls' | 'puts';
}

interface OptionsFlowProps {
  symbol: string;
}

export const OptionsFlow = ({ symbol }: OptionsFlowProps) => {
  const [flowData, setFlowData] = useState<OptionsFlowData[]>([]);
  const [putCallRatio, setPutCallRatio] = useState(0.85);
  const [unusualActivity, setUnusualActivity] = useState([
    { strike: 180, type: 'calls', volume: 15000, premium: 2.45, unusual: true },
    { strike: 175, type: 'puts', volume: 12000, premium: 1.85, unusual: true },
    { strike: 185, type: 'calls', volume: 8500, premium: 1.25, unusual: false },
  ]);

  useEffect(() => {
    const generateFlowData = () => {
      const data = [];
      const baseStrike = 180;
      
      for (let i = -5; i <= 5; i++) {
        const strike = baseStrike + i * 2.5;
        data.push({
          strike,
          calls: Math.floor(Math.random() * 5000) + 1000,
          puts: Math.floor(Math.random() * 3000) + 500,
          callVolume: Math.floor(Math.random() * 10000) + 2000,
          putVolume: Math.floor(Math.random() * 8000) + 1500,
          oi: Math.floor(Math.random() * 15000) + 5000,
          type: Math.random() > 0.5 ? 'calls' : 'puts'
        });
      }
      return data;
    };

    setFlowData(generateFlowData());
    
    const interval = setInterval(() => {
      setFlowData(generateFlowData());
      setPutCallRatio(0.7 + Math.random() * 0.6);
    }, 4000);

    return () => clearInterval(interval);
  }, [symbol]);

  const pieData = [
    { name: 'Calls', value: 60, color: '#00ff88' },
    { name: 'Puts', value: 40, color: '#ff4757' }
  ];

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">OPTIONS FLOW</h2>
          </div>
          <div className="text-xs text-terminal-muted">{symbol}</div>
        </div>
        
        {/* Put/Call Ratio */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-terminal-muted">P/C Ratio</span>
          <span className={`text-sm font-semibold ${
            putCallRatio > 1 ? 'text-terminal-red' : 'text-terminal-green'
          }`}>
            {putCallRatio.toFixed(2)}
          </span>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        {/* Unusual Activity */}
        <div className="p-2 border-b border-terminal-border/50">
          <div className="text-xs text-terminal-muted mb-2">UNUSUAL ACTIVITY</div>
          <div className="space-y-1">
            {unusualActivity.slice(0, 2).map((activity, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <Circle className={`w-2 h-2 ${activity.unusual ? 'text-terminal-orange' : 'text-terminal-muted'}`} />
                  <span>${activity.strike}</span>
                  <span className={activity.type === 'calls' ? 'text-terminal-green' : 'text-terminal-red'}>
                    {activity.type.toUpperCase()}
                  </span>
                </div>
                <div className="text-terminal-cyan financial-number">
                  {activity.volume.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Options Flow Chart */}
        <div className="flex-1 p-2">
          <div className="grid grid-cols-2 gap-2 h-full">
            {/* Volume Flow */}
            <div>
              <div className="text-xs text-terminal-muted mb-1">Volume Flow</div>
              <ResponsiveContainer width="100%" height="80%">
                <BarChart data={flowData.slice(0, 6)}>
                  <XAxis 
                    dataKey="strike" 
                    axisLine={false}
                    tick={{ fontSize: 9, fill: '#888888' }}
                  />
                  <YAxis hide />
                  <Bar dataKey="callVolume" stackId="a">
                    {flowData.slice(0, 6).map((entry, index) => (
                      <Cell key={`call-${index}`} fill="#00ff88" opacity={0.8} />
                    ))}
                  </Bar>
                  <Bar dataKey="putVolume" stackId="a">
                    {flowData.slice(0, 6).map((entry, index) => (
                      <Cell key={`put-${index}`} fill="#ff4757" opacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Distribution */}
            <div>
              <div className="text-xs text-terminal-muted mb-1">Distribution</div>
              <ResponsiveContainer width="100%" height="80%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={20}
                    outerRadius={40}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        
        {/* Strike Levels */}
        <div className="p-2 border-t border-terminal-border/50">
          <div className="text-xs text-terminal-muted mb-1">KEY STRIKES</div>
          <div className="grid grid-cols-3 gap-1 text-xs">
            <div className="text-center">
              <div className="text-terminal-red">175 PUT</div>
              <div className="text-terminal-muted">12K</div>
            </div>
            <div className="text-center">
              <div className="text-terminal-cyan">180 ATM</div>
              <div className="text-terminal-muted">25K</div>
            </div>
            <div className="text-center">
              <div className="text-terminal-green">185 CALL</div>
              <div className="text-terminal-muted">18K</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
