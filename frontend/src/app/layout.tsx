import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'MorganVuoksi Terminal | Quantitative Trading Platform',
  description: 'Professional-grade quantitative research and trading terminal powered by AI/ML models',
  keywords: 'trading, quantitative finance, bloomberg terminal, AI trading, portfolio optimization',
  authors: [{ name: 'MorganVuoksi Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#ff6b1a',
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="theme-color" content="#ff6b1a" />
        <style jsx global>{`
          :root {
            --font-inter: ${inter.style.fontFamily};
            --font-jetbrains-mono: ${jetbrainsMono.style.fontFamily};
          }
        `}</style>
      </head>
      <body className="bg-terminal-bg text-text-primary antialiased">
        {/* Terminal Chrome */}
        <div className="min-h-screen flex flex-col">
          {/* Top Navigation Bar */}
          <nav className="bg-header-bg border-b-2 border-bloomberg-orange px-4 py-2 flex items-center justify-between shadow-terminal">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-bloomberg-red"></div>
                <div className="w-3 h-3 rounded-full bg-bloomberg-amber"></div>
                <div className="w-3 h-3 rounded-full bg-bloomberg-green"></div>
              </div>
              <div className="flex items-center space-x-2">
                <h1 className="text-bloomberg-orange font-bold text-lg tracking-wide">
                  MORGANVUOKSI
                </h1>
                <span className="text-text-secondary text-sm font-mono">
                  TERMINAL
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm font-mono">
                <span className="text-text-secondary">STATUS:</span>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 rounded-full bg-bloomberg-green animate-pulse"></div>
                  <span className="text-bloomberg-green">ONLINE</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-2 text-sm font-mono">
                <span className="text-text-secondary">TIME:</span>
                <span className="text-text-primary" id="terminal-time">
                  {new Date().toLocaleTimeString()}
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                <button className="btn-secondary text-xs px-3 py-1">
                  SETTINGS
                </button>
                <button className="btn-primary text-xs px-3 py-1">
                  HELP
                </button>
              </div>
            </div>
          </nav>

          {/* Main Terminal Content */}
          <main className="flex-1 flex">
            {/* Left Sidebar - Watchlist/Navigation */}
            <aside className="w-64 bg-surface-primary border-r border-border-secondary p-4 terminal-scrollbar overflow-y-auto">
              <div className="space-y-4">
                <div className="terminal-card">
                  <div className="bg-surface-secondary px-3 py-2 border-b border-border-secondary">
                    <h3 className="text-bloomberg-orange text-sm font-semibold uppercase tracking-wide">
                      Quick Access
                    </h3>
                  </div>
                  <div className="p-3 space-y-2">
                    {['Markets', 'Portfolio', 'Research', 'Options', 'News', 'Analytics'].map((item) => (
                      <button
                        key={item}
                        className="w-full text-left px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-surface-hover rounded transition-all"
                      >
                        {item.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="terminal-card">
                  <div className="bg-surface-secondary px-3 py-2 border-b border-border-secondary">
                    <h3 className="text-bloomberg-orange text-sm font-semibold uppercase tracking-wide">
                      Watchlist
                    </h3>
                  </div>
                  <div className="p-3 space-y-1 font-mono text-xs">
                    {[
                      { symbol: 'AAPL', price: '175.43', change: '+1.23', pct: '+0.71%', positive: true },
                      { symbol: 'GOOGL', price: '136.47', change: '+0.59', pct: '+0.43%', positive: true },
                      { symbol: 'TSLA', price: '242.68', change: '-2.14', pct: '-0.87%', positive: false },
                      { symbol: 'MSFT', price: '378.85', change: '+4.21', pct: '+1.12%', positive: true },
                      { symbol: 'NVDA', price: '785.32', change: '-12.45', pct: '-1.56%', positive: false },
                    ].map((stock) => (
                      <div key={stock.symbol} className="flex justify-between items-center py-1 px-2 hover:bg-surface-hover rounded">
                        <span className="text-text-primary font-semibold">{stock.symbol}</span>
                        <div className="text-right">
                          <div className="text-text-primary">{stock.price}</div>
                          <div className={stock.positive ? 'price-positive' : 'price-negative'}>
                            {stock.change} ({stock.pct})
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </aside>

            {/* Center Content Area */}
            <section className="flex-1 bg-bloomberg-black">
              {children}
            </section>

            {/* Right Sidebar - News/Info */}
            <aside className="w-80 bg-surface-primary border-l border-border-secondary p-4 terminal-scrollbar overflow-y-auto">
              <div className="space-y-4">
                <div className="terminal-card">
                  <div className="bg-surface-secondary px-3 py-2 border-b border-border-secondary">
                    <h3 className="text-bloomberg-orange text-sm font-semibold uppercase tracking-wide">
                      Market News
                    </h3>
                  </div>
                  <div className="p-3 space-y-3">
                    {[
                      {
                        headline: 'Fed Signals Continued Rate Hikes',
                        time: '2 min ago',
                        source: 'Reuters',
                      },
                      {
                        headline: 'Tech Stocks Rally on AI Optimism',
                        time: '15 min ago',
                        source: 'Bloomberg',
                      },
                      {
                        headline: 'Energy Sector Shows Strong Growth',
                        time: '32 min ago',
                        source: 'WSJ',
                      },
                      {
                        headline: 'Crypto Market Volatility Continues',
                        time: '1 hr ago',
                        source: 'CoinDesk',
                      },
                    ].map((news, idx) => (
                      <div key={idx} className="border-b border-border-secondary pb-2 last:border-b-0 last:pb-0">
                        <h4 className="text-text-primary text-sm font-medium mb-1 leading-tight">
                          {news.headline}
                        </h4>
                        <div className="flex justify-between text-xs">
                          <span className="text-bloomberg-orange">{news.source}</span>
                          <span className="text-text-muted">{news.time}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="terminal-card">
                  <div className="bg-surface-secondary px-3 py-2 border-b border-border-secondary">
                    <h3 className="text-bloomberg-orange text-sm font-semibold uppercase tracking-wide">
                      Market Indices
                    </h3>
                  </div>
                  <div className="p-3 space-y-2 font-mono text-sm">
                    {[
                      { name: 'S&P 500', value: '4,567.89', change: '+23.45', pct: '+0.52%', positive: true },
                      { name: 'NASDAQ', value: '14,234.56', change: '+87.23', pct: '+0.62%', positive: true },
                      { name: 'DOW', value: '35,678.12', change: '-45.67', pct: '-0.13%', positive: false },
                      { name: 'VIX', value: '18.45', change: '-1.23', pct: '-6.25%', positive: false },
                    ].map((index) => (
                      <div key={index.name} className="flex justify-between items-center">
                        <span className="text-text-secondary">{index.name}</span>
                        <div className="text-right">
                          <div className="text-text-primary">{index.value}</div>
                          <div className={index.positive ? 'price-positive' : 'price-negative'}>
                            {index.change} ({index.pct})
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </aside>
          </main>

          {/* Bottom Status Bar */}
          <footer className="bg-surface-secondary border-t border-border-secondary px-4 py-2 flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4 font-mono">
              <span className="text-text-secondary">DATA:</span>
              <span className="text-bloomberg-green">LIVE</span>
              <span className="text-text-secondary">•</span>
              <span className="text-text-secondary">LATENCY:</span>
              <span className="text-bloomberg-amber">12ms</span>
              <span className="text-text-secondary">•</span>
              <span className="text-text-secondary">MEMORY:</span>
              <span className="text-text-primary">1.2GB</span>
            </div>
            
            <div className="flex items-center space-x-4 font-mono text-xs">
              <span className="text-text-muted">© 2024 MorganVuoksi Terminal</span>
              <span className="text-bloomberg-orange">v2.1.0</span>
            </div>
          </footer>
        </div>

        {/* Real-time Clock Script */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              function updateClock() {
                const now = new Date();
                const timeElement = document.getElementById('terminal-time');
                if (timeElement) {
                  timeElement.textContent = now.toLocaleTimeString();
                }
              }
              
              // Update clock every second
              setInterval(updateClock, 1000);
              
              // Initial update
              updateClock();
            `,
          }}
        />
      </body>
    </html>
  )
}