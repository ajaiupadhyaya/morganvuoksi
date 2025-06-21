"use client";

import DataTable from "@/components/DataTable";
import Headlines from "@/components/Headlines";
import KeyExecutives from "@/components/KeyExecutives";
import PrimaryChart from "@/components/PrimaryChart";
import Watchlist from "@/components/Watchlist";
import { TerminalData } from "@/types";
import { useEffect, useState } from "react";

export default function Home() {
  const [data, setData] = useState<TerminalData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/v1/terminal_data", {
          cache: "no-store",
        });
        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status} ${res.statusText}`);
        }
        const terminalData: TerminalData = await res.json();
        setData(terminalData);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unknown error occurred"
        );
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh data every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-primary">
        <p className="text-accent-red">Error: {error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-primary">
        <p className="animate-pulse text-secondary-foreground">
          Loading Terminal Data...
        </p>
      </div>
    );
  }

  const { symbol, watchlist, price_chart, key_executives, headlines } = data;
  const isPositiveChange = symbol.change_pct >= 0;

  return (
    <div className="flex h-screen w-full flex-col font-sans text-primary-foreground bg-primary">
      <header className="flex h-14 flex-shrink-0 items-center justify-between border-b border-muted/50 px-6">
        <div className="flex items-center gap-4">
          <p className="font-mono text-xl font-bold text-accent-blue">MV</p>
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-semibold">{symbol.ticker}</h1>
            <p className="text-sm text-muted-foreground">{symbol.name}</p>
          </div>
        </div>
        <div className="flex items-baseline gap-4 font-mono text-lg">
          <p>{symbol.price.toFixed(2)}</p>
          <p
            className={isPositiveChange ? "text-accent-green" : "text-accent-red"}
          >
            {isPositiveChange ? "+" : ""}
            {symbol.change_val.toFixed(2)}
          </p>
          <p
            className={isPositiveChange ? "text-accent-green" : "text-accent-red"}
          >
            ({isPositiveChange ? "+" : ""}
            {symbol.change_pct.toFixed(2)}%)
          </p>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-12 gap-6 p-6">
        <aside className="col-span-2 flex flex-col gap-6">
          <Watchlist items={watchlist} />
        </aside>

        <div className="col-span-7 flex flex-col gap-6">
          <div className="h-[60%]">
            <PrimaryChart data={price_chart["1Y"]} />
          </div>
          <div className="h-[40%]">
            <DataTable />
          </div>
        </div>

        <aside className="col-span-3 flex flex-col gap-6">
          <KeyExecutives executives={key_executives} />
          <Headlines headlines={headlines} />
        </aside>
      </main>
    </div>
  );
}
