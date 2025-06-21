"use client";
import { FC } from "react";
import Card from "./ui/Card";
import { BookOpen } from "lucide-react";

// Mock data for now, will be passed as props later
const trades = [
  { symbol: "AAPL", type: "Call", strike: 150, premium: 2.5, quantity: 10 },
  { symbol: "GOOGL", type: "Put", strike: 2800, premium: 15.2, quantity: 5 },
  { symbol: "TSLA", type: "Call", strike: 700, premium: 8.1, quantity: 12 },
  { symbol: "MSFT", type: "Put", strike: 300, premium: 5.5, quantity: 8 },
  { symbol: "AMZN", type: "Call", strike: 140, premium: 3.8, quantity: 15 },
];

const DataTable: FC = () => {
  return (
    <Card>
      <h2 className="mb-4 flex items-center gap-2 font-semibold text-secondary-foreground">
        <BookOpen size={18} />
        Recent Trades
      </h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm text-left">
          <thead className="text-muted-foreground">
            <tr>
              <th className="p-2">Symbol</th>
              <th className="p-2">Type</th>
              <th className="p-2 text-right">Strike</th>
              <th className="p-2 text-right">Premium</th>
              <th className="p-2 text-right">Quantity</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade, index) => (
              <tr
                key={index}
                className="border-t border-muted/50 hover:bg-secondary/40"
              >
                <td className="p-2 font-medium">{trade.symbol}</td>
                <td className="p-2">{trade.type}</td>
                <td className="p-2 text-right font-mono">
                  {trade.strike.toFixed(2)}
                </td>
                <td className="p-2 text-right font-mono">
                  {trade.premium.toFixed(2)}
                </td>
                <td className="p-2 text-right font-mono">{trade.quantity}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};

export default DataTable; 