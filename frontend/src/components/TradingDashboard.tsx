"use client";

import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, BarChart3, AlertTriangle, Settings, Zap, BookOpen, Target, Brain, PieChart, Search, Terminal, Bell, Layout, Globe, Shield, CheckCircle, Clock, Volume2 } from 'lucide-react';
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
import { Watchlist } from './Watchlist';

const TradingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('market');
  const [data, setData] = useState<TerminalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [layout, setLayout] = useState('professional');
  const [notifications, setNotifications] = useState(3);
  const [commandInput, setCommandInput] = useState('');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connected');
  const [systemStats, setSystemStats] = useState({
    cpu: 12,
    memory: 2.1,
    latency: 0.8,
    uptime: '4d 12h 23m'
  });

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

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      // Update system stats
      setSystemStats(prev => ({
        ...prev,
        cpu: Math.max(5, Math.min(95, prev.cpu + (Math.random() - 0.5) * 5)),
        memory: Math.max(1.0, Math.min(8.0, prev.memory + (Math.random() - 0.5) * 0.2)),
        latency: Math.max(0.1, Math.min(5.0, prev.latency + (Math.random() - 0.5) * 0.3)),
      }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
      // Bloomberg-style function keys
      if (e.key === 'F8') {
        e.preventDefault();
        setSelectedSymbol('AAPL');
        console.log('F8: Equity mode activated');
      }
      if (e.key === 'F9') {
        e.preventDefault();
        console.log('F9: Government bonds activated');
      }
      if (e.key === 'F10') {
        e.preventDefault();
        console.log('F10: Currency activated');
      }
      if (e.key === 'F11') {
        e.preventDefault();
        console.log('F11: Commodities activated');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleCommandSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (commandInput.trim()) {
      // Parse Bloomberg-style commands
      const cmd = commandInput.toUpperCase();
      if (cmd.includes('EQUITY') || cmd.includes('<EQUITY>')) {
        const symbol = cmd.split(' ')[0];
        setSelectedSymbol(symbol);
      }
      console.log('Command executed:', commandInput);
      setCommandInput('');
    }
  };

  const layouts = [
    { id: 'professional', name: 'Professional', icon: Layout },
    { id: 'analytical', name: 'Analytical', icon: BarChart3 },
    { id: 'compact', name: 'Ultra Dense', icon: Volume2 },
    { id: 'trading', name: 'Trading Floor', icon: Activity },
  ];

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
    <div className="min-h-screen bg-terminal-bg text-terminal-text font-mono overflow-hidden">
      {/* ENHANCED BLOOMBERG TERMINAL HEADER */}
      <div className="h-20 bloomberg-header flex flex-col shadow-2xl">
        {/* Top Header Bar - Professional Terminal Style */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-terminal-border/50 bg-gradient-to-r from-terminal-panel/80 to-terminal-bg">
          <div className="flex items-center space-x-6">
            {/* Terminal Logo & Status */}
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Terminal className="w-7 h-7 text-terminal-orange" />
                <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-terminal-green' : 
                  connectionStatus === 'connecting' ? 'bg-terminal-amber' : 'bg-terminal-red'
                } terminal-pulse`} />
              </div>
              <div>
                <div className="bloomberg-logo">
                  BLOOMBERG TERMINAL
                </div>
                <div className="text-xs text-terminal-muted font-mono">
                  PROFESSIONAL EDITION v12.8.4 | USER: TRADER001
                </div>
              </div>
            </div>
            
            {/* Live Market Status */}
            <div className="flex items-center space-x-4 text-xs">
              <div className="flex items-center space-x-2 px-3 py-1 bg-terminal-panel border border-terminal-green">
                <div className="status-indicator live" />
                <span className="text-terminal-green font-bold font-mono">LIVE MARKET DATA</span>
              </div>
              
              <div className="flex items-center space-x-2 text-terminal-muted">
                <Globe className="w-3 h-3" />
                <span className="font-mono">{currentTime.toLocaleTimeString()} EST</span>
              </div>
            </div>
            
            {/* Major Indices */}
            <div className="flex items-center space-x-4 text-xs">
              <div className="flex items-center space-x-1">
                <span className="text-terminal-cyan font-mono font-bold">SPX:</span>
                <span className="text-terminal-text font-mono tabular-nums">4,521.23</span>
                <span className="status-positive font-mono">+12.45</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className="text-terminal-cyan font-mono font-bold">NDX:</span>
                <span className="text-terminal-text font-mono tabular-nums">15,245.67</span>
                <span className="status-negative font-mono">-8.23</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className="text-terminal-cyan font-mono font-bold">VIX:</span>
                <span className="text-terminal-text font-mono tabular-nums">18.42</span>
                <span className="status-negative font-mono">-0.68</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Function Key Shortcuts */}
            <div className="flex items-center space-x-1 text-xs text-terminal-muted border-l border-terminal-border pl-3">
              <kbd className="function-key">F8</kbd>
              <span>EQUITIES</span>
              <kbd className="function-key">F9</kbd>
              <span>BONDS</span>
              <kbd className="function-key">F10</kbd>
              <span>FX</span>
              <kbd className="function-key">F11</kbd>
              <span>CMDTY</span>
            </div>

            {/* System Status */}
            <div className="flex items-center space-x-3 text-xs border-l border-terminal-border pl-3">
              <div className="flex items-center space-x-1">
                <Cpu className="w-3 h-3 text-terminal-muted" />
                <span className="text-terminal-cyan font-mono">{systemStats.cpu}%</span>
              </div>
              <div className="flex items-center space-x-1">
                <MemoryStick className="w-3 h-3 text-terminal-muted" />
                <span className="text-terminal-cyan font-mono">{systemStats.memory}GB</span>
              </div>
              <div className="flex items-center space-x-1">
                <Signal className="w-3 h-3 text-terminal-muted" />
                <span className="status-positive font-mono">{systemStats.latency}ms</span>
              </div>
            </div>

            {/* Connection Status */}
            <div className="flex items-center space-x-1">
              {connectionStatus === 'connected' ? (
                <Wifi className="w-4 h-4 text-terminal-green" />
              ) : connectionStatus === 'connecting' ? (
                <Wifi className="w-4 h-4 text-terminal-amber animate-pulse" />
              ) : (
                <WifiOff className="w-4 h-4 text-terminal-red" />
              )}
            </div>

            {/* Layout Selector */}
            <select 
              className="terminal-input text-xs"
              value={layout}
              onChange={(e) => setLayout(e.target.value)}
            >
              {layouts.map((layoutOption) => (
                <option key={layoutOption.id} value={layoutOption.id}>
                  {layoutOption.name}
                </option>
              ))}
            </select>

            {/* Notifications */}
            <button className="terminal-button relative">
              <Bell className="w-3 h-3" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-4 h-4 flex items-center justify-center text-white font-bold animate-pulse">
                  {notifications}
                </span>
              )}
            </button>

            {/* Settings */}
            <button className="terminal-button">
              <Settings className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Bloomberg Command Line - Professional */}
        <div className="h-8 flex items-center px-4 command-line">
          <div className="flex items-center space-x-2 w-full">
            <div className="flex items-center space-x-2">
              <Command className="w-3 h-3 text-terminal-orange" />
              <span className="text-terminal-orange font-mono text-xs font-bold">COMMAND:</span>
            </div>
            <form onSubmit={handleCommandSubmit} className="flex-1 flex items-center">
              <ChevronRight className="w-3 h-3 text-terminal-cyan mr-1" />
              <input
                type="text"
                value={commandInput}
                onChange={(e) => setCommandInput(e.target.value)}
                placeholder="Enter Bloomberg command (e.g., AAPL <Equity> GP, SPX <Index> HP, NEWS <GO>)..."
                className="bg-transparent text-terminal-text placeholder-terminal-muted text-xs font-mono w-full focus:outline-none border-none tracking-wide"
              />
            </form>
            <div className="flex items-center space-x-2">
              <button 
                onClick={() => setShowCommandPalette(true)}
                className="flex items-center space-x-1 text-xs text-terminal-muted hover:text-terminal-orange transition-colors"
              >
                <Search className="w-3 h-3" />
                <kbd className="function-key">⌘K</kbd>
              </button>
              <div className="text-xs text-terminal-muted">
                <span className="status-positive">●</span> READY
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ULTRA-DENSE PROFESSIONAL DASHBOARD GRID */}
      <div className="grid grid-cols-20 gap-0.5 p-0.5 h-[calc(100vh-80px-24px)] overflow-hidden dense-layout">
        {/* Market Overview - Top Strip */}
        <div className="col-span-20 h-20">
          <MarketOverview />
        </div>

        {/* Left Column - Watchlist & Portfolio */}
        <div className="col-span-3 space-y-0.5 overflow-hidden">
          <div className="h-[40%]">
            <Watchlist 
              selectedSymbol={selectedSymbol}
              onSymbolSelect={setSelectedSymbol}
            />
          </div>
          <div className="h-[32%]">
            <PortfolioSummary />
          </div>
          <div className="h-[26%]">
            <RiskManager />
          </div>
        </div>

        {/* Center-Left Column - Primary Chart */}
        <div className="col-span-7 space-y-0.5">
          <div className="h-[70%]">
            <PriceChart symbol={selectedSymbol} />
          </div>
          <div className="h-[28%]">
            <TechnicalIndicators symbol={selectedSymbol} />
          </div>
        </div>

        {/* Center Column - AI Analysis */}
        <div className="col-span-4 space-y-0.5">
          <div className="h-[28%]">
            <AIMarketAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[24%]">
            <div className="terminal-panel h-full p-2">
              <div className="text-terminal-orange font-mono font-bold text-xs mb-2 uppercase">
                SENTIMENT ANALYSIS
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-terminal-muted">BULLISH:</span>
                  <span className="status-positive">68%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-terminal-muted">BEARISH:</span>
                  <span className="status-negative">22%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-terminal-muted">NEUTRAL:</span>
                  <span className="status-neutral">10%</span>
                </div>
              </div>
            </div>
          </div>
          <div className="h-[24%]">
            <OptionsFlow symbol={selectedSymbol} />
          </div>
          <div className="h-[22%]">
            <div className="terminal-panel h-full p-2">
              <div className="text-terminal-orange font-mono font-bold text-xs mb-2 uppercase">
                SCREENER
              </div>
              <div className="space-y-1 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-terminal-cyan">RSI &gt; 70:</span>
                  <span className="text-terminal-text">23</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-cyan">RSI &lt; 30:</span>
                  <span className="text-terminal-text">8</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-cyan">NEW HIGHS:</span>
                  <span className="text-terminal-text">156</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Center-Right Column - Order Book & Trading */}
        <div className="col-span-3 space-y-0.5">
          <div className="h-[45%]">
            <OrderBook symbol={selectedSymbol} />
          </div>
          <div className="h-[53%]">
            <div className="terminal-panel h-full p-2">
              <div className="text-terminal-orange font-mono font-bold text-xs mb-2 uppercase">
                TRADING INTERFACE
              </div>
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-1 text-xs">
                  <button className="terminal-button status-positive">BUY</button>
                  <button className="terminal-button status-negative">SELL</button>
                </div>
                <input 
                  type="number" 
                  placeholder="Quantity"
                  className="terminal-input w-full"
                />
                <input 
                  type="number" 
                  placeholder="Price"
                  className="terminal-input w-full"
                />
                <select className="terminal-input w-full">
                  <option>MARKET</option>
                  <option>LIMIT</option>
                  <option>STOP</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - News & Alerts */}
        <div className="col-span-3 space-y-0.5">
          <div className="h-[100%]">
            <div className="terminal-panel h-full p-2 overflow-y-auto">
              <div className="text-terminal-orange font-mono font-bold text-xs mb-2 uppercase">
                NEWS & ALERTS
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="border-l-2 border-terminal-green pl-2">
                  <div className="text-terminal-green font-bold">MARKET OPEN</div>
                  <div className="text-terminal-muted">Markets opened higher on positive earnings</div>
                  <div className="text-terminal-muted ultra-dense">09:30 EST</div>
                </div>
                <div className="border-l-2 border-terminal-red pl-2">
                  <div className="text-terminal-red font-bold">VOLATILITY ALERT</div>
                  <div className="text-terminal-muted">VIX spiked above 20</div>
                  <div className="text-terminal-muted ultra-dense">09:15 EST</div>
                </div>
                <div className="border-l-2 border-terminal-cyan pl-2">
                  <div className="text-terminal-cyan font-bold">EARNINGS</div>
                  <div className="text-terminal-muted">AAPL reports after close</div>
                  <div className="text-terminal-muted ultra-dense">08:45 EST</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Professional Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 h-6 bg-terminal-panel border-t-2 border-terminal-border flex items-center justify-between px-4 text-xs text-terminal-muted ultra-dense">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <CheckCircle className="w-3 h-3 text-terminal-green" />
            <span className="font-mono">MARKET DATA FEED: ACTIVE</span>
          </div>
          <span className="font-mono">LAST UPDATE: {currentTime.toLocaleTimeString()}</span>
          <span className="font-mono">SESSION: {systemStats.uptime}</span>
        </div>
        
        <div className="flex items-center space-x-4">
          <span className="font-mono">CPU: {systemStats.cpu}%</span>
          <span className="font-mono">MEM: {systemStats.memory}GB</span>
          <span className="font-mono">LAT: {systemStats.latency}ms</span>
          <div className="flex items-center space-x-1">
            <Signal className="w-3 h-3 text-terminal-green" />
            <span className="font-mono text-terminal-green">CONNECTED</span>
          </div>
        </div>
      </div>

      {/* Enhanced Professional Floating Actions */}
      <div className="fixed bottom-8 right-4 flex flex-col space-y-1">
        <button className="bg-terminal-orange hover:bg-terminal-amber text-terminal-bg p-2 border border-terminal-orange glow-orange transition-all duration-300 hover:scale-105 group relative">
          <Brain className="w-4 h-4" />
          <div className="absolute right-full mr-2 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap border border-terminal-border">
            AI ANALYSIS ENGINE
          </div>
        </button>
        <button className="bg-terminal-cyan hover:bg-terminal-cyan/80 text-terminal-bg p-2 border border-terminal-cyan glow-cyan transition-all duration-300 hover:scale-105 group relative">
          <Target className="w-4 h-4" />
          <div className="absolute right-full mr-2 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap border border-terminal-border">
            INSTANT EXECUTION
          </div>
        </button>
        <button className="bg-terminal-green hover:bg-terminal-green/80 text-terminal-bg p-2 border border-terminal-green transition-all duration-300 hover:scale-105 group relative">
          <Activity className="w-4 h-4" />
          <div className="absolute right-full mr-2 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap border border-terminal-border">
            LIVE ACTIVITY FEED
          </div>
        </button>
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