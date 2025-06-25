
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
  PieChart, Activity, Target, Layers, Monitor, Command
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
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState(3);
  const [commandInput, setCommandInput] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
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
    { id: 'compact', name: 'Compact', icon: Monitor },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-terminal-bg via-terminal-bg to-terminal-panel text-terminal-text font-mono">
      {/* Enhanced Bloomberg-style Header */}
      <div className="h-16 bg-gradient-to-r from-terminal-panel via-terminal-panel to-terminal-bg border-b-2 border-terminal-orange flex flex-col shadow-lg backdrop-blur-sm">
        {/* Top Header Bar */}
        <div className="h-10 flex items-center justify-between px-6 border-b border-terminal-border/50">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Terminal className="w-6 h-6 text-terminal-orange animate-pulse-subtle" />
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
              </div>
              <div>
                <span className="font-bold text-xl text-terminal-orange bg-gradient-to-r from-terminal-orange to-terminal-amber bg-clip-text text-transparent">
                  BLOOMBERG TERMINAL
                </span>
                <div className="text-xs text-terminal-muted">Professional Edition v12.8.4</div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
                <span className="text-terminal-green font-medium">LIVE MARKET DATA</span>
              </div>
              <div className="text-terminal-muted">
                {currentTime.toLocaleTimeString()} EST
              </div>
              <div className="text-terminal-cyan font-mono">
                SPX: 4,521.23 <span className="text-terminal-green">+12.45</span>
              </div>
              <div className="text-terminal-cyan font-mono">
                NDX: 15,245.67 <span className="text-terminal-red">-8.23</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Function Key Shortcuts Display */}
            <div className="flex items-center space-x-2 text-xs text-terminal-muted">
              <kbd className="bg-terminal-border px-2 py-1 rounded">F8</kbd>
              <span>Equity</span>
              <kbd className="bg-terminal-border px-2 py-1 rounded">F9</kbd>
              <span>Bonds</span>
            </div>

            {/* Layout Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button flex items-center space-x-2 hover:glow-orange">
                <Layout className="w-4 h-4" />
                <span className="hidden md:inline">Layout</span>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel>Choose Layout</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {layouts.map((layoutOption) => (
                  <DropdownMenuItem 
                    key={layoutOption.id}
                    onClick={() => setLayout(layoutOption.id)}
                    className="hover:bg-terminal-border"
                  >
                    <layoutOption.icon className="w-4 h-4 mr-2" />
                    {layoutOption.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Notifications */}
            <button className="terminal-button relative hover:glow-orange">
              <Bell className="w-4 h-4" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 bg-terminal-red text-xs rounded-full w-5 h-5 flex items-center justify-center text-white font-bold animate-pulse">
                  {notifications}
                </span>
              )}
            </button>

            {/* Settings */}
            <DropdownMenu>
              <DropdownMenuTrigger className="terminal-button hover:glow-cyan">
                <Settings className="w-4 h-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-terminal-panel border-terminal-border">
                <DropdownMenuLabel>Terminal Settings</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Shield className="w-4 h-4 mr-2" />
                  Security Settings
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Monitor className="w-4 h-4 mr-2" />
                  Display Preferences
                </DropdownMenuItem>
                <DropdownMenuItem className="hover:bg-terminal-border">
                  <Zap className="w-4 h-4 mr-2" />
                  Performance Tuning
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Bloomberg-style Command Line */}
        <div className="h-6 flex items-center px-6 bg-gradient-to-r from-terminal-bg to-terminal-panel border-b border-terminal-border/30">
          <div className="flex items-center space-x-2 w-full">
            <Command className="w-3 h-3 text-terminal-orange" />
            <form onSubmit={handleCommandSubmit} className="flex-1">
              <input
                type="text"
                value={commandInput}
                onChange={(e) => setCommandInput(e.target.value)}
                placeholder="Enter command (e.g., AAPL <Equity> GP for price graph)..."
                className="bg-transparent text-terminal-text placeholder-terminal-muted text-xs font-mono w-full focus:outline-none border-none"
              />
            </form>
            <button 
              onClick={() => setShowCommandPalette(true)}
              className="flex items-center space-x-1 text-xs text-terminal-muted hover:text-terminal-orange transition-colors"
            >
              <Search className="w-3 h-3" />
              <kbd className="bg-terminal-border px-1 rounded text-xs">⌘K</kbd>
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Main Dashboard Grid */}
      <div className="grid grid-cols-16 gap-1 p-1 h-[calc(100vh-64px)] overflow-hidden">
        {/* Market Overview - Top Row */}
        <div className="col-span-16 h-20">
          <MarketOverview />
        </div>

        {/* Left Column - Watchlist & Portfolio */}
        <div className="col-span-3 space-y-1 overflow-hidden">
          <div className="h-[40%]">
            <Watchlist 
              selectedSymbol={selectedSymbol}
              onSymbolSelect={setSelectedSymbol}
            />
          </div>
          <div className="h-[35%]">
            <PortfolioSummary />
          </div>
          <div className="h-[23%]">
            <RiskManager />
          </div>
        </div>

        {/* Center-Left Column - Charts & Technical Analysis */}
        <div className="col-span-6 space-y-1">
          <div className="h-[65%]">
            <PriceChart symbol={selectedSymbol} />
          </div>
          <div className="h-[33%]">
            <TechnicalIndicators symbol={selectedSymbol} />
          </div>
        </div>

        {/* Center-Right Column - AI Analysis & Sentiment */}
        <div className="col-span-4 space-y-1">
          <div className="h-[30%]">
            <AIMarketAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[25%]">
            <SentimentAnalysis symbol={selectedSymbol} />
          </div>
          <div className="h-[22%]">
            <OptionsFlow symbol={selectedSymbol} />
          </div>
          <div className="h-[21%]">
            <AdvancedScreener />
          </div>
        </div>

        {/* Right Column - Trading & Order Management */}
        <div className="col-span-3 space-y-1">
          <div className="h-[45%]">
            <TradingInterface symbol={selectedSymbol} />
          </div>
          <div className="h-[30%]">
            <OrderBook symbol={selectedSymbol} />
          </div>
          <div className="h-[23%]">
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

      {/* Bloomberg-style Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 h-6 bg-terminal-panel border-t border-terminal-border flex items-center justify-between px-4 text-xs text-terminal-muted">
        <div className="flex items-center space-x-4">
          <span>Connected to Market Data Feed</span>
          <span className="text-terminal-green">●</span>
          <span>Last Update: {currentTime.toLocaleTimeString()}</span>
        </div>
        <div className="flex items-center space-x-4">
          <span>CPU: 12%</span>
          <span>Memory: 2.1GB</span>
          <span>Latency: 0.8ms</span>
        </div>
      </div>

      {/* Enhanced Floating Action Panel */}
      <div className="fixed bottom-8 right-4 flex flex-col space-y-2">
        <button className="bg-terminal-orange hover:bg-terminal-amber text-terminal-bg p-3 rounded-full shadow-2xl glow-orange transition-all duration-300 hover:scale-110 group">
          <Brain className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            AI Analysis
          </div>
        </button>
        <button className="bg-terminal-cyan hover:bg-terminal-cyan/80 text-terminal-bg p-3 rounded-full shadow-2xl glow-cyan transition-all duration-300 hover:scale-110 group">
          <Target className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Quick Trade
          </div>
        </button>
        <button className="bg-terminal-green hover:bg-terminal-green/80 text-terminal-bg p-3 rounded-full shadow-2xl transition-all duration-300 hover:scale-110 group">
          <Activity className="w-5 h-5" />
          <div className="absolute right-full mr-3 top-1/2 transform -translate-y-1/2 bg-terminal-panel px-2 py-1 rounded text-xs text-terminal-text opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Live Activity
          </div>
        </button>
      </div>
    </div>
  );
};
