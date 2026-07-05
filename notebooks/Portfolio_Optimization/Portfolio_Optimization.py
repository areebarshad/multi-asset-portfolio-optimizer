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

# === TANGENCY PORTFOLIO (MAX SHARPE) ===

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

portfolio_return = np.dot(mu, optimal_weights)
portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(sigma, optimal_weights)))
print(f"Expected Portfolio Return: {portfolio_return:.4f}")
print(f"Expected Portfolio Volatility: {portfolio_volatility:.4f}")

#plot portfolio weights
plt.figure(figsize = (10, 6))
plt.bar(daily_returns.columns, optimal_weights)
plt.title("Optimal Portfolio Weights (Tangency)")
plt.ylabel("Weight")
plt.xticks(rotation = 45)
plt.show()

# === EFFICIENT FRONTIER ===

#generate efficient frontier (min-variance sweep for frontier curve)
target_returns = np.linspace(0.02, 0.12, 20)

portfolio_vols = []
portfolio_returns = []
frontier_weights = []

for r in target_returns:
  W = cp.Variable(n)
  objective = cp.Minimize(cp.quad_form(W, sigma))
  constraints = [cp.sum(W) == 1, W >= 0, W <= 0.20, mu @ W >= r]
  prob = cp.Problem(objective, constraints)
  prob.solve()

  if W.value is not None:
    portfolio_vols.append(np.sqrt(np.dot(W.value.T, np.dot(sigma, W.value))))
    portfolio_returns.append(np.dot(mu, W.value))
    frontier_weights.append(W.value)
  else:
    portfolio_vols.append(None)
    portfolio_returns.append(None)
    frontier_weights.append(None)

portfolio_vols = [np.nan if v is None else v for v in portfolio_vols]
portfolio_returns = [np.nan if r is None else r for r in portfolio_returns]
portfolio_vols = np.array(portfolio_vols)
portfolio_returns = np.array(portfolio_returns)
valid = ~np.isnan(portfolio_vols)

# Methodology fix: net frontier subtracts 10bps transaction cost per unit turnover vs equal-weight baseline
c = 0.001
eq_w_baseline = np.ones(n) / n
net_returns = np.array([
    portfolio_returns[i] - c * np.sum(np.abs(frontier_weights[i] - eq_w_baseline))
    if frontier_weights[i] is not None else np.nan
    for i in range(len(portfolio_returns))
])
net_valid = ~np.isnan(net_returns)

# Tangency point and Capital Market Line (reuse already-solved weights)
tangency_weights = optimal_weights_original
tangency_vol = np.sqrt(tangency_weights @ sigma @ tangency_weights)
tangency_ret = mu @ tangency_weights
cml_slope = (tangency_ret - rf) / tangency_vol
cml_vols = np.linspace(0, np.nanmax(portfolio_vols) * 1.2, 100)
cml_rets = rf + cml_slope * cml_vols

#plot efficient frontier with gross/net curves, tangency point, and CML
plt.figure(figsize = (10, 6))
plt.plot(portfolio_vols[valid], portfolio_returns[valid], 'o-', label='Gross frontier')
plt.plot(portfolio_vols[net_valid], net_returns[net_valid], 's--', label='Net frontier (10bps cost)')
plt.plot(cml_vols, cml_rets, 'k--', alpha=0.6, label='Capital Market Line')
plt.plot(tangency_vol, tangency_ret, '*', color='gold', markersize=15, label='Tangency portfolio (max Sharpe)')
plt.axhline(y=rf, color='gray', linestyle=':', alpha=0.5)
plt.annotate(f'rf={rf}', xy=(0, rf), xytext=(0.005, rf + 0.002), fontsize=8, color='gray')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Returns")
plt.title("Efficient Frontier")
plt.legend()
plt.show()
