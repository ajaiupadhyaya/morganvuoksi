export interface TerminalData {
  symbol: {
    name: string;
    ticker: string;
    price: number;
    change_val: number;
    change_pct: number;
    volume: string;
    market_cap: string;
  };
  price_chart: {
    "1D": PriceDataPoint[];
    "5D": PriceDataPoint[];
    "1M": PriceDataPoint[];
    "1Y": PriceDataPoint[];
  };
  watchlist: WatchlistItem[];
  headlines: Headline[];
  key_executives: KeyExecutive[];
}

export interface PriceDataPoint {
  time: string;
  price: number;
}

export interface WatchlistItem {
  ticker: string;
  price: number;
  change_pct: number;
}

export interface Headline {
  source: string;
  title: string;
}

export interface KeyExecutive {
  name: string;
  title: string;
} 