import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# === DATA SETUP ===
# Re-download data so this script runs standalone without CSV dependencies.

#define multi-asset tickers (15-asset expanded universe)
tickers = ['VEA', 'VWO', 'EWJ', 'TLT', 'LQD', 'GLD', 'VNQ', 'DBC',
           'SPY', 'QQQ', 'IEF', 'HYG', 'SLV', 'GSG', 'EMLC']

#download the closing prices
data = yf.download(tickers, start = "2015-01-01", end = "2025-12-31")['Close']

# Fix: deprecated fillna(method=) syntax replaced with chained ffill/bfill
data = data.ffill().bfill()

#calculate daily log returns
daily_returns = np.log(data / data.shift(1)).dropna()

# === OPTIMIZATION SETUP ===

#compute annualized mean returns
mean_daily_returns = daily_returns.mean()
annual_returns = mean_daily_returns * 252

#calculate covariance
cov_matrix = daily_returns.cov()
annual_cov_matrix = cov_matrix * 252

#convert to NumPy arrays for optimization
mu = annual_returns.values
sigma = annual_cov_matrix.values

#number of assets
n = len(mu)

# Risk-free rate (4% approximation for 2015-2025)
rf = 0.04

# Tangency portfolio: maximizes Sharpe ratio via variable substitution
# clip assets at rf+0.001 to prevent infeasibility when mu[i] <= rf
mu_clipped = np.maximum(mu, rf + 0.001)
y = cp.Variable(n)
k = cp.Variable()
objective = cp.Minimize(cp.quad_form(y, sigma))
constraints = [
    (mu_clipped - rf) @ y == 1,
    cp.sum(y) == k,
    y >= 0,
    y <= 0.20 * k,
    k >= 0
]
problem = cp.Problem(objective, constraints)
problem.solve()
print("Problem status:", problem.status)

# Fix: store before w is overwritten by turnover-constrained optimization below
if y.value is not None and k.value is not None and k.value > 1e-8:
    optimal_weights_original = y.value / k.value
else:
    print("WARNING: Original tangency optimization failed")
    optimal_weights_original = np.ones(n) / n

optimal_weights = optimal_weights_original

# === BACKTESTING ===

#ensure dimensions match
assert len(optimal_weights) == daily_returns.shape[1], "Weight length does not match number of assets."

#calculate daily portfolio returns
portfolio_daily_returns = daily_returns @ optimal_weights

#calculate cumulative returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod()

#plot cumulative returns
plt.figure(figsize = (12, 6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.title("Cumulative Returns: Optimized Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#calculate equal-weight portfolio returns
equal_weight = np.repeat(1 / daily_returns.shape[1], daily_returns.shape[1])
equal_weight_returns = daily_returns @ equal_weight
equal_weight_cum = (1 + equal_weight_returns).cumprod()

#plot with equal-weight returns
plt.figure(figsize = (12, 6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.plot(equal_weight_cum, label = "Equal-Weight Portfolio")
plt.title("Cumulative Returns: Optimized and Equal-Weight Portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

#download benchmark (SPY) data
benchmark = yf.download('SPY', start = '2015-01-01', end = '2025-12-31')['Close']

#calculate the benchmark returns
benchmark_returns = np.log(benchmark / benchmark.shift(1)).dropna()
benchmark_cum = (1 + benchmark_returns).cumprod()

#plot the comparison
plt.figure(figsize = (12,6))
plt.plot(cumulative_returns, label = "Optimized Portfolio")
plt.plot(equal_weight_cum, label = "Equal-Weight Portfolio")
plt.plot(benchmark_cum, label = "SPY Benchmark")
plt.title("Cumulative Returns VS Benchmark")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

# === PERFORMANCE METRICS ===

#calculate the performance metrics
def performance_metrics(returns, name = "Portfolio"):
  if isinstance(returns, pd.DataFrame):
    returns = returns.squeeze()

  ann_return = float(returns.mean() * 252)
  ann_vol = float(returns.std() * np.sqrt(252))
  # Fix: subtract rf (4% risk-free rate approximation for 2015-2025) before dividing by volatility
  sharpe_ratio = (ann_return - rf) / ann_vol
  max_dd = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()

  print(f"{name} Performance Metrics")
  print(f"Annual Return: {ann_return:.4f}")
  print(f"Annual Volatility: {ann_vol:.4f}")
  print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
  print(f"Max Drawdown: {max_dd:.4%}\n")

#optimized portfolio
performance_metrics(portfolio_daily_returns, "Optimized Portfolio")

#equal-weight portfolio
performance_metrics(equal_weight_returns, "Equal-Weight Portfolio")

#SPY benchmark portfolio
performance_metrics(benchmark_returns, "SPY Benchmark")

#print robust optimal weights
print("Robust Optimal Weights: ")
for i in range(n):
  print(f"{daily_returns.columns[i]}: {optimal_weights[i]:.4f}")
