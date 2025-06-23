'use client'

import { useState, useEffect } from 'react'
import PrimaryChart from '@/components/PrimaryChart'
import DataTable from '@/components/DataTable'
import Headlines from '@/components/Headlines'
import KeyExecutives from '@/components/KeyExecutives'

export default function TerminalHomepage() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL')
  const [marketData, setMarketData] = useState({
    price: 175.43,
    change: 1.23,
    changePercent: 0.71,
    volume: 45782634,
    marketCap: '2.74T',
    pe: 28.47,
    high52: 198.23,
    low52: 124.17,
  })

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => ({
        ...prev,
        price: prev.price + (Math.random() - 0.5) * 2,
        change: prev.change + (Math.random() - 0.5) * 0.1,
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const tabs = [
    'Overview', 'Chart', 'News', 'Financials', 'Options', 'Research', 'ESG'
  ]

  const [activeTab, setActiveTab] = useState('Overview')

  return (
    <div className="h-full flex flex-col">
      {/* Terminal Header */}
      <div className="bg-header-bg border-b-2 border-bloomberg-orange p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                className="terminal-input w-24 text-center text-lg font-bold"
                placeholder="SYMBOL"
              />
              <span className="text-text-secondary text-sm">US Equity</span>
            </div>
            
            <div className="flex items-center space-x-6 font-mono">
              <div className="text-3xl font-bold text-text-primary">
                ${marketData.price.toFixed(2)}
              </div>
              <div className={`text-lg ${marketData.change >= 0 ? 'price-positive' : 'price-negative'}`}>
                {marketData.change >= 0 ? '+' : ''}{marketData.change.toFixed(2)} 
                ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
              </div>
              <div className="text-sm text-text-secondary">
                Vol: {marketData.volume.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="grid grid-cols-2 gap-4 text-sm font-mono">
              <div>
                <span className="text-text-secondary">Market Cap:</span>
                <span className="text-text-primary ml-2">{marketData.marketCap}</span>
              </div>
              <div>
                <span className="text-text-secondary">P/E:</span>
                <span className="text-text-primary ml-2">{marketData.pe}</span>
              </div>
              <div>
                <span className="text-text-secondary">52W High:</span>
                <span className="text-text-primary ml-2">${marketData.high52}</span>
              </div>
              <div>
                <span className="text-text-secondary">52W Low:</span>
                <span className="text-text-primary ml-2">${marketData.low52}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="nav-tabs mt-4">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`nav-tab ${activeTab === tab ? 'active' : ''}`}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="flex-1 p-4 grid grid-cols-12 gap-4 terminal-scrollbar overflow-auto">
        {/* Primary Chart */}
        <div className="col-span-8 terminal-card">
          <div className="card-header flex items-center justify-between">
            <h3>Price Chart | {selectedSymbol}</h3>
            <div className="flex items-center space-x-2 text-xs">
              <button className="btn-ghost px-2 py-1">1D</button>
              <button className="btn-ghost px-2 py-1">5D</button>
              <button className="btn-ghost px-2 py-1">1M</button>
              <button className="btn-primary px-2 py-1">3M</button>
              <button className="btn-ghost px-2 py-1">1Y</button>
            </div>
          </div>
          <div className="card-content h-96">
            <PrimaryChart symbol={selectedSymbol} />
          </div>
        </div>

        {/* Key Metrics */}
        <div className="col-span-4 space-y-4">
          <div className="terminal-card">
            <div className="card-header">
              <h3>Key Statistics</h3>
            </div>
            <div className="card-content">
              <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                {[
                  { label: 'Open', value: '$174.22' },
                  { label: 'High', value: '$176.45' },
                  { label: 'Low', value: '$173.88' },
                  { label: 'Close', value: '$175.43' },
                  { label: 'Beta', value: '1.24' },
                  { label: 'EPS (TTM)', value: '$6.16' },
                  { label: 'Dividend', value: '$0.96' },
                  { label: 'Yield', value: '0.55%' },
                ].map((item) => (
                  <div key={item.label} className="flex justify-between">
                    <span className="text-text-secondary">{item.label}:</span>
                    <span className="text-text-primary">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="terminal-card">
            <div className="card-header">
              <h3>Trading Activity</h3>
            </div>
            <div className="card-content">
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-text-secondary">Avg Volume (10d):</span>
                  <span className="text-text-primary font-mono">52.3M</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-text-secondary">Volume:</span>
                  <span className="text-bloomberg-orange font-mono">45.8M</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-text-secondary">Shares Outstanding:</span>
                  <span className="text-text-primary font-mono">15.6B</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-text-secondary">Float:</span>
                  <span className="text-text-primary font-mono">15.5B</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Financial Data Table */}
        <div className="col-span-8 terminal-card">
          <div className="card-header">
            <h3>Financial Data</h3>
          </div>
          <div className="card-content">
            <DataTable symbol={selectedSymbol} />
          </div>
        </div>

        {/* Recent News */}
        <div className="col-span-4 terminal-card">
          <div className="card-header">
            <h3>Recent Headlines</h3>
          </div>
          <div className="card-content">
            <Headlines symbol={selectedSymbol} />
          </div>
        </div>

        {/* Technical Indicators */}
        <div className="col-span-6 terminal-card">
          <div className="card-header">
            <h3>Technical Indicators</h3>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-3 gap-4 text-sm font-mono">
              {[
                { name: 'RSI (14)', value: '67.23', signal: 'Neutral', color: 'text-bloomberg-amber' },
                { name: 'MACD', value: '0.42', signal: 'Buy', color: 'price-positive' },
                { name: 'MA (50)', value: '$168.45', signal: 'Above', color: 'price-positive' },
                { name: 'MA (200)', value: '$162.78', signal: 'Above', color: 'price-positive' },
                { name: 'Bollinger', value: 'Upper', signal: 'Sell', color: 'price-negative' },
                { name: 'Williams %R', value: '-23.45', signal: 'Buy', color: 'price-positive' },
              ].map((indicator) => (
                <div key={indicator.name} className="text-center p-3 bg-surface-secondary rounded">
                  <div className="text-text-secondary text-xs mb-1">{indicator.name}</div>
                  <div className="text-text-primary font-semibold">{indicator.value}</div>
                  <div className={`text-xs ${indicator.color}`}>{indicator.signal}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Key Executives */}
        <div className="col-span-6 terminal-card">
          <div className="card-header">
            <h3>Key Executives</h3>
          </div>
          <div className="card-content">
            <KeyExecutives symbol={selectedSymbol} />
          </div>
        </div>

        {/* AI Analysis Panel */}
        <div className="col-span-12 terminal-card">
          <div className="card-header flex items-center justify-between">
            <h3>AI Market Analysis</h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-bloomberg-green animate-pulse"></div>
              <span className="text-xs text-bloomberg-green">AI ACTIVE</span>
            </div>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-surface-secondary p-4 rounded-lg">
                <h4 className="text-bloomberg-orange text-sm font-semibold mb-2">SENTIMENT ANALYSIS</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Bullish:</span>
                    <span className="price-positive">67%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Bearish:</span>
                    <span className="price-negative">23%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Neutral:</span>
                    <span className="text-text-muted">10%</span>
                  </div>
                </div>
              </div>

              <div className="bg-surface-secondary p-4 rounded-lg">
                <h4 className="text-bloomberg-orange text-sm font-semibold mb-2">PRICE TARGET</h4>
                <div className="space-y-2">
                  <div className="text-2xl font-bold text-text-primary">$189.50</div>
                  <div className="text-sm price-positive">+8.02% upside</div>
                  <div className="text-xs text-text-muted">Based on ML ensemble model</div>
                </div>
              </div>

              <div className="bg-surface-secondary p-4 rounded-lg">
                <h4 className="text-bloomberg-orange text-sm font-semibold mb-2">RISK ASSESSMENT</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">VaR (1d, 95%):</span>
                    <span className="text-text-warning">-2.3%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Beta:</span>
                    <span className="text-text-primary">1.24</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Volatility:</span>
                    <span className="text-text-primary">24.7%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}