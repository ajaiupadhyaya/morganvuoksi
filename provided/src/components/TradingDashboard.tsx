import React, { useState, useEffect } from 'react';
import { MarketOverview } from './MarketOverview';
import { PriceChart } from './PriceChart';
import { Watchlist } from './Watchlist';
import { NewsFeed } from './NewsFeed';
import { OrderBook } from './OrderBook';
import { PortfolioSummary } from './PortfolioSummary';
import { CommandPalette } from './CommandPalette';
import { TechnicalIndicators } from './TechnicalIndicators';
import { AIMarketAnalysis } from './AIMarketAnalysis';
import { SentimentAnalysis } from './SentimentAnalysis';
import { OptionsFlow } from './OptionsFlow';
import { TradingInterface } from './TradingInterface';
import { AdvancedScreener } from './AdvancedScreener';
import { RiskManager } from './RiskManager';
import { 
  Search, Terminal, BarChart, Wallet, Bell, Settings, 
  Layout, Brain, TrendingUp, Globe, Shield, Zap,
  PieChart, Activity, Target, Layers, Monitor, Command,
  WifiOff, Wifi, ChevronRight, Database, Cpu, MemoryStick,
  Signal, AlertTriangle, CheckCircle
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";

export const TradingDashboard = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [layout, setLayout] = useState('professional');
  const [theme, setTheme] = useState('terminal');
  const [notifications, setNotifications] = useState(3);
  const [commandInput, setCommandInput] = useState('');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connected');
  const [systemStats, setSystemStats] = useState({
    cpu: 12,
    memory: 2.1,
    latency: 0.8,
    uptime: '4d 12h 23m'
  });

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
    { id: 'analytical', name: 'Analytical', icon: BarChart },
    { id: 'compact', name: 'Ultra Dense', icon: Monitor },
    { id: 'trading', name: 'Trading Floor', icon: Activity },
  ];

  return (
    <div className="min-h-screen bg-terminal-bg text-terminal-text font-mono overflow-hidden">
      {/* ENHANCED BLOOMBERG TERMINAL HEADER */}
      <div className="h-20 bg-gradient-to-r from-terminal-bg via-terminal-panel to-terminal-bg border-b-2 border-terminal-orange flex flex-col shadow-2xl">
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
                <div className="font-bold text-xl text-terminal-orange tracking-widest">
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
                <div className="w-2 h-2 bg-terminal-green rounded-full terminal-pulse" />
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
                <span className="text-terminal-green font-mono">+12.45</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className="text-terminal-cyan font-mono font-bold">NDX:</span>
                <span className="text-terminal-text font-mono tabular-nums">15,245.67</span>
                <span className="text-terminal-red font-mono">-8.23</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className="text-terminal-cyan font-mono font-bold">VIX:</span>
                <span className="text-terminal-text font-mono tabular-nums">18.42</span>
                <span className="text-terminal-red font-mono">-0.68</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Function Key Shortcuts */}
            <div className="flex items-center space-x-1 text-xs text-terminal-muted border-l border-terminal-border pl-3">
              <kbd className="bg-terminal-border px-1.5 py-0.5 text-xs font-mono">F8</kbd>
              <span>EQUITIES</span>
              <kbd className="bg-terminal-border px-1.5 py-0.5 text-xs font-mono">F9</kbd>
              <span>BONDS</span>
              <kbd className="bg-terminal-border px-1.5 py-0.5 text-xs font-mono">F10</kbd>
              <span>FX</span>
              <kbd className="bg-terminal-border px-1.5 py-0.5 text-xs font-mono">F11</kbd>
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
                <span className="text-terminal-green font-mono">{systemStats.latency}ms</span>
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
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button flex items-center space-x-1 hover:glow-orange">
                <Layout className="w-3 h-3" />
                <span className="text-xs">LAYOUT</span>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel className="text-terminal-cyan">TERMINAL LAYOUT</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {layouts.map((layoutOption) => (
                  <DropdownMenuItem 
                    key={layoutOption.id}
                    onClick={() => setLayout(layoutOption.id)}
                    className="hover:bg-terminal-border text-terminal-text"
                  >
                    <layoutOption.icon className="w-3 h-3 mr-2" />
                    {layoutOption.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Notifications */}
            <button className="terminal-button relative hover:glow-orange">
              <Bell className="w-3 h-3" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-4 h-4 flex items-center justify-center text-white font-bold animate-pulse">
                  {notifications}
                </span>
              )}
            </button>

            {/* Settings */}
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button hover:glow-cyan">
                <Settings className="w-3 h-3" />
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel className="text-terminal-cyan">TERMINAL CONFIG</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="hover:bg-terminal-border text-terminal-text">
                  <Shield className="w-3 h-3 mr-2" />
                  Security Settings
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border text-terminal-text">
                  <Monitor className="w-3 h-3 mr-2" />
                  Display Config
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border text-terminal-text">
                  <Database className="w-3 h-3 mr-2" />
                  Data Feeds
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Bloomberg Command Line - Professional */}
        <div className="h-8 flex items-center px-4 bg-gradient-to-r from-terminal-bg to-terminal-panel border-b border-terminal-border/30 command-line">
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
                <kbd className="bg-terminal-border px-1 text-xs">⌘K</kbd>
              </button>
              <div className="text-xs text-terminal-muted">
                <span className="text-terminal-green">●</span> READY
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
            <SentimentAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[24%]">
            <OptionsFlow symbol={selectedSymbol} />
          </div>
          <div className="h-[22%]">
            <AdvancedScreener />
          </div>
        </div>

        {/* Center-Right Column - Order Book & Trading */}
        <div className="col-span-3 space-y-0.5">
          <div className="h-[45%]">
            <OrderBook symbol={selectedSymbol} />
          </div>
          <div className="h-[53%]">
            <TradingInterface symbol={selectedSymbol} />
          </div>
        </div>

        {/* Right Column - News & Alerts */}
        <div className="col-span-3 space-y-0.5">
          <div className="h-[100%]">
            <NewsFeed />
          </div>
        </div>
      </div>

      {/* Command Palette */}
      {showCommandPalette && (
        <CommandPalette 
          onClose={() => setShowCommandPalette(false)}
          onSymbolSelect={setSelectedSymbol}
        />
      )}

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
