#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class BacktestEngine {
private:
    double initial_capital;
    double transaction_cost;
    double slippage;
    double risk_free_rate;
    double position_size;
    double stop_loss;
    double take_profit;

    struct Trade {
        int entry_date;
        int exit_date;
        double entry_price;
        double exit_price;
        double position;
        double pnl;
    };

    struct Portfolio {
        std::vector<double> position;
        std::vector<double> cash;
        std::vector<double> holdings;
        std::vector<double> total;
        std::vector<double> returns;
        std::vector<Trade> trades;
    };

public:
    BacktestEngine(double initial_capital, double transaction_cost, double slippage,
                  double risk_free_rate, double position_size, double stop_loss,
                  double take_profit)
        : initial_capital(initial_capital), transaction_cost(transaction_cost),
          slippage(slippage), risk_free_rate(risk_free_rate),
          position_size(position_size), stop_loss(stop_loss),
          take_profit(take_profit) {}

    Portfolio run_backtest(const py::array_t<double>& prices,
                          const py::array_t<double>& signals) {
        auto prices_r = prices.unchecked<2>();
        auto signals_r = signals.unchecked<1>();
        
        Portfolio portfolio;
        int n = prices_r.shape(0);
        
        // Initialize portfolio
        portfolio.position.resize(n, 0);
        portfolio.cash.resize(n, initial_capital);
        portfolio.holdings.resize(n, 0);
        portfolio.total.resize(n, initial_capital);
        portfolio.returns.resize(n, 0);
        
        double current_position = 0;
        double entry_price = 0;
        
        for (int i = 1; i < n; i++) {
            // Get current price with slippage
            double price = prices_r(i, 3);  // Close price
            if (signals_r(i) > 0) {
                price *= (1 + slippage);
            } else if (signals_r(i) < 0) {
                price *= (1 - slippage);
            }
            
            // Update position
            if (signals_r(i) != 0 && signals_r(i) != current_position) {
                // Close existing position
                if (current_position != 0) {
                    double exit_price = price;
                    double pnl = (exit_price - entry_price) * current_position;
                    portfolio.cash[i] += pnl;
                    
                    Trade trade;
                    trade.entry_date = i - 1;
                    trade.exit_date = i;
                    trade.entry_price = entry_price;
                    trade.exit_price = exit_price;
                    trade.position = current_position;
                    trade.pnl = pnl;
                    portfolio.trades.push_back(trade);
                }
                
                // Open new position
                current_position = signals_r(i);
                entry_price = price;
                portfolio.position[i] = current_position;
                
                // Apply transaction costs
                double cost = std::abs(current_position) * price * transaction_cost;
                portfolio.cash[i] -= cost;
            }
            
            // Update portfolio
            portfolio.holdings[i] = current_position * price;
            portfolio.total[i] = portfolio.cash[i] + portfolio.holdings[i];
            portfolio.returns[i] = portfolio.total[i] / portfolio.total[i-1] - 1;
        }
        
        return portfolio;
    }

    std::map<std::string, double> calculate_metrics(const Portfolio& portfolio) {
        std::map<std::string, double> metrics;
        
        // Calculate returns
        double total_return = (portfolio.total.back() / initial_capital) - 1;
        double annual_return = std::pow(1 + total_return, 252.0 / portfolio.returns.size()) - 1;
        
        // Calculate volatility
        double mean_return = std::accumulate(portfolio.returns.begin(),
                                           portfolio.returns.end(), 0.0) / portfolio.returns.size();
        double variance = 0;
        for (double r : portfolio.returns) {
            variance += std::pow(r - mean_return, 2);
        }
        variance /= portfolio.returns.size();
        double volatility = std::sqrt(variance * 252);
        
        // Calculate Sharpe ratio
        double sharpe_ratio = (annual_return - risk_free_rate) / volatility;
        
        // Calculate Sortino ratio
        double downside_variance = 0;
        int downside_count = 0;
        for (double r : portfolio.returns) {
            if (r < 0) {
                downside_variance += std::pow(r, 2);
                downside_count++;
            }
        }
        downside_variance /= downside_count;
        double downside_volatility = std::sqrt(downside_variance * 252);
        double sortino_ratio = (annual_return - risk_free_rate) / downside_volatility;
        
        // Calculate maximum drawdown
        double max_drawdown = 0;
        double peak = portfolio.total[0];
        for (double value : portfolio.total) {
            if (value > peak) {
                peak = value;
            }
            double drawdown = (peak - value) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
        
        // Calculate win rate
        int winning_trades = 0;
        for (const Trade& trade : portfolio.trades) {
            if (trade.pnl > 0) {
                winning_trades++;
            }
        }
        double win_rate = static_cast<double>(winning_trades) / portfolio.trades.size();
        
        // Store metrics
        metrics["total_return"] = total_return;
        metrics["annual_return"] = annual_return;
        metrics["volatility"] = volatility;
        metrics["sharpe_ratio"] = sharpe_ratio;
        metrics["sortino_ratio"] = sortino_ratio;
        metrics["max_drawdown"] = max_drawdown;
        metrics["win_rate"] = win_rate;
        metrics["num_trades"] = portfolio.trades.size();
        
        return metrics;
    }
};

PYBIND11_MODULE(backtest_engine, m) {
    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<double, double, double, double, double, double, double>())
        .def("run_backtest", &BacktestEngine::run_backtest)
        .def("calculate_metrics", &BacktestEngine::calculate_metrics);
} 