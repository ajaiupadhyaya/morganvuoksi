"use client";

import React from 'react';
import { Newspaper, TrendingUp, TrendingDown, Clock, ExternalLink, Heart, MessageCircle } from 'lucide-react';
import { TerminalData } from '@/types';

interface NewsAndSentimentProps {
  data: TerminalData;
}

const NewsAndSentiment: React.FC<NewsAndSentimentProps> = ({ data }) => {
  const { headlines } = data;

  // Mock sentiment data - in real implementation, this would come from your NLP analysis
  const sentimentData = {
    overall: 0.65, // Positive sentiment score
    bullish: 68,
    neutral: 24,
    bearish: 8,
    volume: 1247
  };

  return (
    <div className="h-full p-6 space-y-6">
      {/* Sentiment Overview */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Heart className="w-4 h-4" />
            MARKET SENTIMENT
          </h3>
          <div className="flex items-center gap-2">
            <span className="status-indicator status-live"></span>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">REAL-TIME</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* Overall Sentiment */}
          <div className="metric-card">
            <div className="metric-label">SENTIMENT SCORE</div>
            <div className="metric-value text-[var(--bloomberg-terminal-green)]">
              {(sentimentData.overall * 100).toFixed(0)}
            </div>
            <div className="metric-change positive-change">
              <TrendingUp className="w-3 h-3 inline mr-1" />
              BULLISH
            </div>
          </div>

          {/* Bullish */}
          <div className="metric-card">
            <div className="metric-label">BULLISH</div>
            <div className="metric-value text-[var(--gains-color)]">{sentimentData.bullish}%</div>
          </div>

          {/* Neutral */}
          <div className="metric-card">
            <div className="metric-label">NEUTRAL</div>
            <div className="metric-value text-[var(--neutral-color)]">{sentimentData.neutral}%</div>
          </div>

          {/* Bearish */}
          <div className="metric-card">
            <div className="metric-label">BEARISH</div>
            <div className="metric-value text-[var(--losses-color)]">{sentimentData.bearish}%</div>
          </div>

          {/* Volume */}
          <div className="metric-card">
            <div className="metric-label">NEWS VOLUME</div>
            <div className="metric-value">{sentimentData.volume.toLocaleString()}</div>
            <div className="metric-change text-[var(--bloomberg-text-secondary)]">
              <MessageCircle className="w-3 h-3 inline mr-1" />
              24H
            </div>
          </div>
        </div>

        {/* Sentiment Bar */}
        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">SENTIMENT DISTRIBUTION</span>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">100%</span>
          </div>
          <div className="w-full bg-[var(--bloomberg-surface)] rounded-full h-3 overflow-hidden">
            <div className="h-full flex">
              <div 
                className="bg-[var(--gains-color)] h-full transition-all duration-500"
                style={{ width: `${sentimentData.bullish}%` }}
              ></div>
              <div 
                className="bg-[var(--neutral-color)] h-full transition-all duration-500"
                style={{ width: `${sentimentData.neutral}%` }}
              ></div>
              <div 
                className="bg-[var(--losses-color)] h-full transition-all duration-500"
                style={{ width: `${sentimentData.bearish}%` }}
              ></div>
            </div>
          </div>
          <div className="flex justify-between mt-1 text-xs font-mono">
            <span className="text-[var(--gains-color)]">Bullish {sentimentData.bullish}%</span>
            <span className="text-[var(--neutral-color)]">Neutral {sentimentData.neutral}%</span>
            <span className="text-[var(--losses-color)]">Bearish {sentimentData.bearish}%</span>
          </div>
        </div>
      </div>

      {/* News Feed */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title flex items-center gap-2">
            <Newspaper className="w-4 h-4" />
            FINANCIAL NEWS
          </h3>
          <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
            {headlines.length} ARTICLES
          </span>
        </div>

        <div className="space-y-4 max-h-96 overflow-y-auto">
          {headlines.map((headline, index) => {
            // Mock sentiment for each headline
            const headlineSentiment = Math.random();
            const sentimentColor = headlineSentiment > 0.6 ? 'text-[var(--gains-color)]' : 
                                  headlineSentiment < 0.4 ? 'text-[var(--losses-color)]' : 
                                  'text-[var(--neutral-color)]';
            const sentimentIcon = headlineSentiment > 0.6 ? TrendingUp : 
                                 headlineSentiment < 0.4 ? TrendingDown : 
                                 MessageCircle;
            const SentimentIcon = sentimentIcon;

            return (
              <div 
                key={index}
                className="p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)] hover:bg-[var(--bloomberg-hover)] transition-colors group"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-[var(--bloomberg-blue)]">
                      {headline.source.toUpperCase()}
                    </span>
                    <span className="text-xs text-[var(--bloomberg-text-secondary)]">•</span>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-[var(--bloomberg-text-secondary)]" />
                      <span className="text-xs text-[var(--bloomberg-text-secondary)]">
                        {Math.floor(Math.random() * 60)} min ago
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`flex items-center gap-1 ${sentimentColor}`}>
                      <SentimentIcon className="w-3 h-3" />
                      <span className="text-xs font-mono">
                        {(headlineSentiment * 100).toFixed(0)}%
                      </span>
                    </div>
                    <ExternalLink className="w-3 h-3 text-[var(--bloomberg-text-secondary)] opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer" />
                  </div>
                </div>
                
                <h4 className="text-sm font-medium leading-snug text-[var(--bloomberg-text-primary)] group-hover:text-[var(--bloomberg-blue)] transition-colors cursor-pointer">
                  {headline.title}
                </h4>

                {/* Mock tags */}
                <div className="flex items-center gap-2 mt-3">
                  {['EARNINGS', 'ANALYST', 'UPGRADE'].slice(0, Math.floor(Math.random() * 3) + 1).map((tag, tagIndex) => (
                    <span 
                      key={tagIndex}
                      className="px-2 py-1 text-xs font-mono bg-[var(--bloomberg-tertiary)] text-[var(--bloomberg-text-secondary)] rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Key Headlines Summary */}
      <div className="bloomberg-card">
        <div className="bloomberg-card-header">
          <h3 className="bloomberg-card-title">KEY MARKET MOVERS</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-[var(--gains-color)]" />
              <span className="font-mono font-bold text-sm text-[var(--gains-color)]">TOP GAINERS</span>
            </div>
            <div className="space-y-2">
              {['TSLA +12.4%', 'NVDA +8.7%', 'AMD +6.2%'].map((gainer, idx) => (
                <div key={idx} className="font-mono text-sm text-[var(--bloomberg-text-secondary)]">
                  {gainer}
                </div>
              ))}
            </div>
          </div>
          
          <div className="p-4 rounded-md bg-[var(--bloomberg-surface)] border border-[var(--bloomberg-border)]">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-[var(--losses-color)]" />
              <span className="font-mono font-bold text-sm text-[var(--losses-color)]">TOP LOSERS</span>
            </div>
            <div className="space-y-2">
              {['META -5.8%', 'NFLX -4.3%', 'PYPL -3.9%'].map((loser, idx) => (
                <div key={idx} className="font-mono text-sm text-[var(--bloomberg-text-secondary)]">
                  {loser}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsAndSentiment;