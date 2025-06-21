import { WatchlistItem } from "@/types";
import { FC } from "react";

interface WatchlistProps {
  items: WatchlistItem[];
}

const Watchlist: FC<WatchlistProps> = ({ items }) => {
  return (
    <div className="h-full rounded-lg bg-secondary/20 p-4">
      <h2 className="mb-4 font-semibold text-secondary-foreground">
        Watchlist
      </h2>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.ticker} className="flex items-baseline justify-between">
            <span className="font-semibold">{item.ticker}</span>
            <div className="font-mono text-right">
              <div>{item.price.toFixed(2)}</div>
              <div
                className={
                  item.change_pct >= 0 ? "text-accent-green" : "text-accent-red"
                }
              >
                {item.change_pct.toFixed(2)}%
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Watchlist; 