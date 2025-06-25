
import React, { useState, useEffect } from 'react';
import { Clock, ExternalLink } from 'lucide-react';

interface NewsItem {
  id: string;
  title: string;
  source: string;
  time: string;
  summary: string;
  relevance: 'high' | 'medium' | 'low';
}

export const NewsFeed = () => {
  const [news, setNews] = useState<NewsItem[]>([
    {
      id: '1',
      title: 'Fed Signals Potential Rate Cuts Ahead',
      source: 'Reuters',
      time: '2h ago',
      summary: 'Federal Reserve officials hint at possible interest rate adjustments in upcoming meetings...',
      relevance: 'high'
    },
    {
      id: '2',
      title: 'Tech Earnings Beat Expectations',
      source: 'Bloomberg',
      time: '4h ago',
      summary: 'Major technology companies report stronger than expected quarterly results...',
      relevance: 'medium'
    },
    {
      id: '3',
      title: 'Oil Prices Rise on Supply Concerns',
      source: 'WSJ',
      time: '6h ago',
      summary: 'Crude oil futures climb as geopolitical tensions raise supply concerns...',
      relevance: 'medium'
    },
    {
      id: '4',
      title: 'Dollar Strengthens Against Euro',
      source: 'Financial Times',
      time: '8h ago',
      summary: 'US Dollar gains ground against major currencies amid economic data...',
      relevance: 'low'
    },
  ]);

  const getRelevanceColor = (relevance: string) => {
    switch (relevance) {
      case 'high': return 'text-terminal-red';
      case 'medium': return 'text-terminal-amber';
      case 'low': return 'text-terminal-cyan';
      default: return 'text-terminal-muted';
    }
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <h2 className="text-sm font-semibold text-terminal-orange">NEWS FEED</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {news.map((item) => (
          <div key={item.id} className="border-b border-terminal-border/30 pb-3 last:border-b-0">
            <div className="flex items-start justify-between mb-2">
              <div className={`w-2 h-2 rounded-full mt-1 ${getRelevanceColor(item.relevance)}`} />
              <div className="flex items-center text-xs text-terminal-muted">
                <Clock className="w-3 h-3 mr-1" />
                {item.time}
              </div>
            </div>
            
            <h3 className="text-sm font-medium text-terminal-text mb-1 hover:text-terminal-cyan cursor-pointer">
              {item.title}
            </h3>
            
            <p className="text-xs text-terminal-muted mb-2 line-clamp-2">
              {item.summary}
            </p>
            
            <div className="flex items-center justify-between">
              <span className="text-xs text-terminal-orange font-medium">
                {item.source}
              </span>
              <ExternalLink className="w-3 h-3 text-terminal-muted hover:text-terminal-cyan cursor-pointer" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
