"""
DCF Valuation Module
Comprehensive discounted cash flow analysis for fundamental valuation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DCFValuator:
    """Advanced DCF valuation model."""
    
    def __init__(self, wacc: float = 0.1, terminal_growth: float = 0.02):
        self.wacc = wacc  # Weighted Average Cost of Capital
        self.terminal_growth = terminal_growth
        
    def calculate_dcf(self, symbol: str, years: int = 5) -> Dict:
        """Calculate DCF valuation for a company."""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            cash_flow = ticker.cashflow
            
            # Extract key financial metrics
            revenue = info.get('totalRevenue', 0)
            free_cash_flow = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 1)
            current_price = info.get('currentPrice', 0)
            
            # Estimate growth rate
            growth_rate = self._estimate_growth_rate(financials, cash_flow)
            
            # Project free cash flows
            projected_fcf = self._project_cash_flows(free_cash_flow, growth_rate, years)
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(projected_fcf[-1])
            
            # Calculate present values
            pv_fcf = self._calculate_present_value(projected_fcf)
            pv_terminal = terminal_value / ((1 + self.wacc) ** years)
            
            # Enterprise and equity value
            enterprise_value = pv_fcf + pv_terminal
            equity_value = enterprise_value  # Simplified - should adjust for net debt
            
            # Per-share intrinsic value
            intrinsic_value = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            
            # Margin of safety
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else 0
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'intrinsic_value': float(intrinsic_value),
                'enterprise_value': float(enterprise_value),
                'equity_value': float(equity_value),
                'margin_of_safety': margin_of_safety,
                'growth_rate': growth_rate,
                'wacc': self.wacc,
                'terminal_growth': self.terminal_growth,
                'projected_fcf': [float(fcf) for fcf in projected_fcf],
                'terminal_value': float(terminal_value),
                'pv_fcf': float(pv_fcf),
                'pv_terminal': float(pv_terminal),
                'recommendation': self._get_recommendation(margin_of_safety),
                'valuation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"DCF calculation error for {symbol}: {str(e)}")
            raise
    
    def _estimate_growth_rate(self, financials: pd.DataFrame, cash_flow: pd.DataFrame) -> float:
        """Estimate growth rate from historical financials."""
        
        try:
            # Try to get revenue growth
            if not financials.empty and 'Total Revenue' in financials.index:
                revenue_series = financials.loc['Total Revenue'].dropna()
                if len(revenue_series) >= 2:
                    # Calculate CAGR
                    years = len(revenue_series) - 1
                    start_revenue = revenue_series.iloc[-1]
                    end_revenue = revenue_series.iloc[0]
                    growth_rate = (end_revenue / start_revenue) ** (1/years) - 1
                    return max(min(growth_rate, 0.20), -0.10)  # Cap between -10% and 20%
            
            # Fallback to industry average
            return 0.05  # 5% default growth
            
        except Exception:
            return 0.05
    
    def _project_cash_flows(self, base_fcf: float, growth_rate: float, years: int) -> List[float]:
        """Project future free cash flows."""
        
        if base_fcf <= 0:
            base_fcf = 1000000  # $1M assumption if no FCF data
        
        projected_fcf = []
        for year in range(1, years + 1):
            # Gradually decline growth rate
            adjusted_growth = growth_rate * (1 - 0.1 * (year - 1))
            fcf = base_fcf * ((1 + adjusted_growth) ** year)
            projected_fcf.append(fcf)
            
        return projected_fcf
    
    def _calculate_terminal_value(self, final_fcf: float) -> float:
        """Calculate terminal value using Gordon Growth Model."""
        
        terminal_fcf = final_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth)
        
        return terminal_value
    
    def _calculate_present_value(self, cash_flows: List[float]) -> float:
        """Calculate present value of projected cash flows."""
        
        pv = 0
        for i, fcf in enumerate(cash_flows, 1):
            pv += fcf / ((1 + self.wacc) ** i)
            
        return pv
    
    def _get_recommendation(self, margin_of_safety: float) -> str:
        """Get investment recommendation based on margin of safety."""
        
        if margin_of_safety > 0.25:
            return "STRONG BUY"
        elif margin_of_safety > 0.10:
            return "BUY"
        elif margin_of_safety > -0.10:
            return "HOLD"
        else:
            return "SELL"
    
    def sensitivity_analysis(self, symbol: str, 
                           wacc_range: Tuple[float, float] = (0.08, 0.12),
                           growth_range: Tuple[float, float] = (0.01, 0.05)) -> Dict:
        """Perform sensitivity analysis on DCF valuation."""
        
        base_dcf = self.calculate_dcf(symbol)
        
        # WACC sensitivity
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], 5)
        wacc_sensitivity = []
        
        for wacc in wacc_values:
            temp_valuator = DCFValuator(wacc=wacc, terminal_growth=self.terminal_growth)
            result = temp_valuator.calculate_dcf(symbol)
            wacc_sensitivity.append({
                'wacc': wacc,
                'intrinsic_value': result['intrinsic_value']
            })
        
        # Growth sensitivity  
        growth_values = np.linspace(growth_range[0], growth_range[1], 5)
        growth_sensitivity = []
        
        for growth in growth_values:
            temp_valuator = DCFValuator(wacc=self.wacc, terminal_growth=growth)
            result = temp_valuator.calculate_dcf(symbol)
            growth_sensitivity.append({
                'terminal_growth': growth,
                'intrinsic_value': result['intrinsic_value']
            })
        
        return {
            'base_valuation': base_dcf,
            'wacc_sensitivity': wacc_sensitivity,
            'growth_sensitivity': growth_sensitivity
        }