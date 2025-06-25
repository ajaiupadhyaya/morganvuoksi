
import React, { useState, useEffect } from 'react';
import { MessageSquare, ThumbsUp, ThumbsDown, TrendingUp, Globe, Twitter } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis } from 'recharts';

interface SentimentData {
  source: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  score: number;
  volume: number;
  keywords: string[];
}

interface SentimentMetrics {
  overall: number;
  bullish: number;
  bearish: number;
  neutral: number;
  confidence: number;
}

export const SentimentAnalysis = ({ symbol }: { symbol: string }) => {
  const [sentimentData, setSentimentData] = useState<SentimentData[]>([]);
  const [metrics, setMetrics] = useState<SentimentMetrics>({
    overall: 0.65,
    bullish: 45,
    bearish: 25,
    neutral: 30,
    confidence: 0.78
  });

  useEffect(() => {
    const generateSentimentData = () => {
      const sources = [
        { name: 'Twitter', icon: Twitter },
        { name: 'Reddit', icon: MessageSquare },
        { name: 'News', icon: Globe },
        { name: 'Forums', icon: MessageSquare }
      ];

      const keywords = [
        'bullish', 'earnings', 'growth', 'innovation', 'breakthrough',
        'bearish', 'correction', 'volatility', 'uncertainty', 'decline'
      ];

      return sources.map(source => {
        const sentiment = Math.random() > 0.4 ? 'positive' : Math.random() > 0.5 ? 'negative' : 'neutral';
        return {
          source: source.name,
          sentiment: sentiment as 'positive' | 'negative' | 'neutral',
          score: Math.random() * 2 - 1, // -1 to 1
          volume: Math.floor(Math.random() * 10000) + 1000,
          keywords: keywords.sort(() => 0.5 - Math.random()).slice(0, 3)
        };
      });
    };

    const updateMetrics = () => {
      const bullish = 30 + Math.random() * 40;
      const bearish = 15 + Math.random() * 30;
      const neutral = 100 - bullish - bearish;
      
      setMetrics({
        overall: (Math.random() * 0.8) + 0.1,
        bullish,
        bearish,
        neutral,
        confidence: 0.7 + Math.random() * 0.25
      });
    };

    setSentimentData(generateSentimentData());
    updateMetrics();

    const interval = setInterval(() => {
      setSentimentData(generateSentimentData());
      updateMetrics();
    }, 8000);

    return () => clearInterval(interval);
  }, [symbol]);

  const pieData = [
    { name: 'Bullish', value: metrics.bullish, color: '#00ff88' },
    { name: 'Bearish', value: metrics.bearish, color: '#ff4757' },
    { name: 'Neutral', value: metrics.neutral, color: '#00d4ff' }
  ];

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-terminal-green';
      case 'negative': return 'text-terminal-red';
      default: return 'text-terminal-cyan';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return ThumbsUp;
      case 'negative': return ThumbsDown;
      default: return TrendingUp;
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">SENTIMENT ANALYSIS</h2>
          </div>
          <div className="text-xs text-terminal-muted">
            AI Confidence: <span className="text-terminal-green">{(metrics.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {/* Overall Sentiment Score */}
        <div className="p-3 border-b border-terminal-border/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-terminal-muted">Overall Sentiment</span>
            <span className={`text-lg font-bold ${
              metrics.overall > 0.6 ? 'text-terminal-green' : 
              metrics.overall < 0.4 ? 'text-terminal-red' : 'text-terminal-cyan'
            }`}>
              {(metrics.overall * 100).toFixed(0)}%
            </span>
          </div>
          <div className="w-full bg-terminal-border rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-1000 ${
                metrics.overall > 0.6 ? 'bg-terminal-green' : 
                metrics.overall < 0.4 ? 'bg-terminal-red' : 'bg-terminal-cyan'
              }`}
              style={{ width: `${metrics.overall * 100}%` }}
            />
          </div>
        </div>

        {/* Sentiment Distribution */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-2">SENTIMENT DISTRIBUTION</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={25}
                  outerRadius={50}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-4 mt-2">
            {pieData.map((item) => (
              <div key={item.name} className="flex items-center space-x-1">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-xs text-terminal-muted">
                  {item.name}: {item.value.toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Source Breakdown */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">SOURCE ANALYSIS</h3>
          <div className="space-y-3">
            {sentimentData.map((item, index) => {
              const IconComponent = getSentimentIcon(item.sentiment);
              return (
                <div key={index} className="bg-terminal-bg/30 rounded p-3 border border-terminal-border/20">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-terminal-text">{item.source}</span>
                      <IconComponent className={`w-3 h-3 ${getSentimentColor(item.sentiment)}`} />
                    </div>
                    <div className="text-xs text-terminal-muted">
                      {item.volume.toLocaleString()} mentions
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-xs font-medium ${getSentimentColor(item.sentiment)}`}>
                      Score: {item.score > 0 ? '+' : ''}{item.score.toFixed(2)}
                    </span>
                  </div>
                  
                  <div className="flex flex-wrap gap-1">
                    {item.keywords.map((keyword, i) => (
                      <span 
                        key={i}
                        className="px-2 py-1 bg-terminal-border/50 text-xs text-terminal-muted rounded"
                      >
                        #{keyword}
                      </span>
                    ))}
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
