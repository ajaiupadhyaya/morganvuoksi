"use client";

import { PriceDataPoint } from "@/types";
import { FC } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";

interface PrimaryChartProps {
  data: PriceDataPoint[];
}

const PriceChart: FC<PrimaryChartProps> = ({ data }) => {
  const formattedData = data.map((item) => ({
    ...item,
    time: new Date(item.time).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  const minPrice = Math.min(...data.map(d => d.price));
  const maxPrice = Math.max(...data.map(d => d.price));
  const avgPrice = data.reduce((sum, d) => sum + d.price, 0) / data.length;

  return (
    <div className="chart-container h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <h4 className="font-mono text-sm font-bold text-[var(--bloomberg-text-primary)]">
            PRICE CHART
          </h4>
          <div className="flex items-center gap-2">
            <span className="status-indicator status-live"></span>
            <span className="text-xs font-mono text-[var(--bloomberg-text-secondary)]">
              REAL-TIME
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-[var(--bloomberg-text-secondary)]">
            MIN: <span className="text-[var(--losses-color)]">${minPrice.toFixed(2)}</span>
          </div>
          <div className="text-[var(--bloomberg-text-secondary)]">
            MAX: <span className="text-[var(--gains-color)]">${maxPrice.toFixed(2)}</span>
          </div>
          <div className="text-[var(--bloomberg-text-secondary)]">
            AVG: <span className="text-[var(--neutral-color)]">${avgPrice.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={formattedData}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
        >
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--bloomberg-terminal-green)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="var(--bloomberg-terminal-green)" stopOpacity={0} />
            </linearGradient>
          </defs>
          
          <XAxis
            dataKey="time"
            stroke="var(--bloomberg-text-secondary)"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            fontFamily="'Roboto Mono', monospace"
          />
          
          <YAxis
            stroke="var(--bloomberg-text-secondary)"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            domain={["dataMin - 1", "dataMax + 1"]}
            fontFamily="'Roboto Mono', monospace"
          />
          
          <ReferenceLine 
            y={avgPrice} 
            stroke="var(--bloomberg-blue)" 
            strokeDasharray="3 3" 
            strokeOpacity={0.7}
          />
          
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--bloomberg-surface)",
              borderColor: "var(--bloomberg-border)",
              color: "var(--bloomberg-text-primary)",
              fontSize: "12px",
              fontFamily: "'Roboto Mono', monospace",
              borderRadius: "6px",
              border: "1px solid var(--bloomberg-border)",
              boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
            }}
            itemStyle={{ 
              color: "var(--bloomberg-terminal-green)",
              fontWeight: "bold"
            }}
            labelStyle={{ 
              color: "var(--bloomberg-text-secondary)",
              fontSize: "11px"
            }}
            formatter={(value) => [`$${value}`, "Price"]}
          />
          
          <Area
            type="monotone"
            dataKey="price"
            stroke="var(--bloomberg-terminal-green)"
            fillOpacity={1}
            fill="url(#priceGradient)"
            strokeWidth={2}
            dot={false}
            activeDot={{
              r: 4,
              fill: "var(--bloomberg-terminal-green)",
              strokeWidth: 2,
              stroke: "var(--bloomberg-surface)"
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;