"use client";

import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, BarChart3, AlertTriangle, Settings, Zap, BookOpen, Target, Brain, PieChart } from 'lucide-react';
import MarketOverview from './MarketOverview';
import PriceChart from './PriceChart';
import NewsAndSentiment from './NewsAndSentiment';
import PortfolioSummary from './PortfolioSummary';
import RiskManager from './RiskManager';
import OptionsFlow from './OptionsFlow';
import AIMarketAnalysis from './AIMarketAnalysis';
import TechnicalIndicators from './TechnicalIndicators';
import OrderBook from './OrderBook';
import { TerminalData } from '@/types';

const TradingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('market');
  const [data, setData] = useState<TerminalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const tabs = [
    { id: 'market', label: 'MARKET DATA', icon: TrendingUp },
    { id: 'charts', label: 'CHARTS', icon: BarChart3 },
    { id: 'portfolio', label: 'PORTFOLIO', icon: PieChart },
    { id: 'ai', label: 'AI ANALYSIS', icon: Brain },
    { id: 'options', label: 'OPTIONS', icon: Target },
    { id: 'risk', label: 'RISK', icon: AlertTriangle },
    { id: 'news', label: 'NEWS', icon: BookOpen },
    { id: 'orders', label: 'ORDERS', icon: Settings },
  ];

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const res = await fetch("http://127.0.0.1:8000/api/v1/terminal_data", {
          cache: "no-store",
        });
        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status} ${res.statusText}`);
        }
        const terminalData: TerminalData = await res.json();
        setData(terminalData);
        setError(null);
        setLastUpdate(new Date());
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const renderTabContent = () => {
    if (isLoading) {
      return <LoadingState />;
    }

    if (error) {
      return <ErrorState error={error} />;
    }

    if (!data) {
      return <EmptyState />;
    }

    switch (activeTab) {
      case 'market':
        return <MarketOverview data={data} />;
      case 'charts':
        return <ChartsTab data={data} />;
      case 'portfolio':
        return <PortfolioSummary data={data} />;
      case 'ai':
        return <AIMarketAnalysis data={data} />;
      case 'options':
        return <OptionsFlow data={data} />;
      case 'risk':
        return <RiskManager data={data} />;
      case 'news':
        return <NewsAndSentiment data={data} />;
      case 'orders':
        return <OrderBook data={data} />;
      default:
        return <MarketOverview data={data} />;
    }
  };

  return (
    <div className="h-screen w-full bg-gradient-to-br from-[var(--bloomberg-primary)] to-[var(--bloomberg-secondary)] text-[var(--bloomberg-text-primary)]">
      {/* Terminal Header */}
      <div className="terminal-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="status-indicator status-live"></span>
              <span className="text-[var(--bloomberg-terminal-green)] font-mono font-bold text-sm">LIVE</span>
            </div>
            <div className="terminal-title">MORGANVUOKSI ELITE TERMINAL</div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-sm font-mono text-[var(--bloomberg-text-primary)]">
                {new Date().toLocaleTimeString()}
              </div>
              <div className="text-xs text-[var(--bloomberg-text-secondary)]">
                Last Update: {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-[var(--bloomberg-terminal-green)]" />
              <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
                {data?.symbol?.ticker || 'AAPL'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-[var(--bloomberg-border)] bg-[var(--bloomberg-surface)]">
        <div className="flex overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 font-mono text-xs font-medium transition-all duration-200 border-b-2 ${
                  activeTab === tab.id
                    ? 'border-[var(--bloomberg-blue)] bg-[var(--bloomberg-tertiary)] text-[var(--bloomberg-blue)]'
                    : 'border-transparent text-[var(--bloomberg-text-secondary)] hover:text-[var(--bloomberg-text-primary)] hover:bg-[var(--bloomberg-hover)]'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {renderTabContent()}
      </div>
    </div>
  );
};

// Loading State Component
const LoadingState: React.FC = () => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center">
      <div className="loading-skeleton w-16 h-16 rounded-full mx-auto mb-4"></div>
      <div className="text-[var(--bloomberg-text-secondary)] font-mono">
        Loading Terminal Data...
      </div>
    </div>
  </div>
);

// Error State Component
const ErrorState: React.FC<{ error: string }> = ({ error }) => (
  <div className="flex items-center justify-center h-full">
    <div className="error-message max-w-md text-center">
      <AlertTriangle className="w-8 h-8 mx-auto mb-4" />
      <div className="font-mono font-bold mb-2">TERMINAL ERROR</div>
      <div className="text-sm">{error}</div>
    </div>
  </div>
);

// Empty State Component
const EmptyState: React.FC = () => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center text-[var(--bloomberg-text-secondary)]">
      <Activity className="w-12 h-12 mx-auto mb-4" />
      <div className="font-mono">No data available</div>
    </div>
  </div>
);

// Charts Tab Component
const ChartsTab: React.FC<{ data: TerminalData }> = ({ data }) => (
  <div className="h-full p-6">
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
      <div className="lg:col-span-3">
        <div className="h-full bloomberg-card">
          <div className="bloomberg-card-header">
            <h3 className="bloomberg-card-title">PRICE CHART</h3>
            <div className="flex items-center gap-2">
              <span className="status-indicator status-live"></span>
              <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
                REAL-TIME
              </span>
            </div>
          </div>
          <div className="h-96">
            <PriceChart data={data.price_chart["1Y"]} />
          </div>
        </div>
      </div>
      
      <div className="lg:col-span-1">
        <TechnicalIndicators data={data} />
      </div>
    </div>
  </div>
);

export default TradingDashboard;