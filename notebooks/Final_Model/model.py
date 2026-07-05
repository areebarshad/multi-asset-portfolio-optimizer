import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.covariance import LedoitWolf

#define multi-asset tickers
tickers = ['VEA', 'VWO', 'EWJ', 'TLT', 'LQD', 'GLD', 'VNQ', 'DBC',
           'SPY', 'QQQ', 'IEF', 'HYG', 'SLV', 'GSG', 'EMLC']

#download the closing prices
data = yf.download(tickers, start = "2015-01-01", end = "2025-12-31")['Close']
print("Data collected and shaped: ", data.shape)
print(data.head())

#check for missing values
print(data.isnull().sum())

# Fix: deprecated fillna(method=) syntax replaced with chained ffill/bfill
data = data.ffill().bfill()

#confirm no missing data remains
print(data.isnull().sum().sum())

#calculate daily returns
daily_returns = np.log(data / data.shift(1)).dropna()

#calculate monthly returns
monthly_prices = data.resample('ME').last()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

#plot the closing prices
plt.figure(figsize = (12, 6))
for ticker in tickers:
  plt.plot(data[ticker], label = ticker)
plt.legend()
plt.title("Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

#plot daily return distributions
daily_returns.plot(kind = 'hist', bins = 100, alpha = 0.7, figsize = (12, 6))
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Log Return")
plt.show()

#plot the correlation heatmap
plt.figure(figsize = (12, 6))
sns.heatmap(daily_returns.corr(), annot = True, cmap = 'coolwarm')
plt.title("Correlation Heatmap of Daily Returns")
plt.show()

#calculate the mean and standard deviation of daily returns
stats = pd.DataFrame({'Mean of Daily Returns': daily_returns.mean(),
                      'Standard Deviation of Daily Returns': daily_returns.std()})

#annualize mean and std (assumption: 252 trading days)
stats['Annualized Return'] = stats['Mean of Daily Returns'] * 252
stats['Annualized Volatility'] = stats["Standard Deviation of Daily Returns"] * np.sqrt(252)

print(stats)

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

# Fix: use solver output instead of hardcoded array
optimal_weights = optimal_weights_original

#ensure dimensions match
assert len(optimal_weights) == daily_returns.shape[1], "Weight length does not match number of assets."

#calculate daily portfolio returns
portfolio_daily_returns = daily_returns @ optimal_weights
# Fix: removed typo variable 'portoflio_daily_returns' and its duplicate assignment

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

#assume previous portfolio weights
prev_weights = np.repeat(1 / n, n)  # equal-weight baseline for 15-asset universe

#define turnover limit
turnover_limit = 0.10

#re-run optimization with turnover constraint (min-variance formulation)
w = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(w, sigma))
constraints = [cp.sum(w) == 1, w >= 0, w <= 0.20,
               cp.norm(w - prev_weights, 1) <= turnover_limit]

problem = cp.Problem(objective, constraints)
problem.solve()

#print optimal weights with turnover constraint
print("Optimal Weights with Turnover Constraint: ")
for i in range(n):
  print(f"{daily_returns.columns[i]}: {w.value[i]:.4f}")


# === WALK-FORWARD BACKTEST ===
wf_returns_list = []
start_year = daily_returns.index.year.min()
end_year = daily_returns.index.year.max()

for train_start_year in range(start_year, end_year - 2):
    train_end_year = train_start_year + 2
    test_year = train_start_year + 3

    if test_year > end_year:
        break

    train_mask = (daily_returns.index.year >= train_start_year) & (daily_returns.index.year <= train_end_year)
    test_mask = (daily_returns.index.year == test_year)

    train_data = daily_returns[train_mask]
    test_data = daily_returns[test_mask]

    if len(train_data) < 100 or len(test_data) < 50:
        continue

    # Walk-forward uses BL+LW expected returns instead of sample mean
    lw_wf = LedoitWolf().fit(train_data)
    sigma_wf = lw_wf.covariance_ * 252

    w_eq_wf = np.ones(n) / n
    delta_ra = 2.5
    Pi_wf = delta_ra * sigma_wf @ w_eq_wf

    P_wf = np.zeros((2, n))
    P_wf[0, 5] = 1    # GLD absolute return view
    P_wf[1, 3] = 1    # TLT absolute return view
    Q_wf = np.array([0.08, 0.02])
    Omega_wf = np.diag([0.0001, 0.0001])

    tau = 0.05
    M_wf = np.linalg.inv(tau * sigma_wf) + P_wf.T @ np.linalg.inv(Omega_wf) @ P_wf
    mu_wf = np.linalg.inv(M_wf) @ (np.linalg.inv(tau * sigma_wf) @ Pi_wf + P_wf.T @ np.linalg.inv(Omega_wf) @ Q_wf)

    # Tangency portfolio: maximizes Sharpe ratio via variable substitution
    mu_wf_clipped = np.maximum(mu_wf, rf + 0.001)
    y_wf = cp.Variable(n)
    k_wf = cp.Variable()
    prob_wf = cp.Problem(
        cp.Minimize(cp.quad_form(y_wf, sigma_wf)),
        [(mu_wf_clipped - rf) @ y_wf == 1, cp.sum(y_wf) == k_wf,
         y_wf >= 0, y_wf <= 0.20 * k_wf, k_wf >= 0]
    )
    prob_wf.solve()

    if y_wf.value is not None and k_wf.value is not None and k_wf.value > 1e-8:
        weights_wf = y_wf.value / k_wf.value
        port_ret = test_data.values @ weights_wf
        wf_returns_list.append(pd.Series(port_ret, index=test_data.index))
    else:
        print(f"WARNING: Walk-forward optimization failed for test year {test_year}")

wf_returns = pd.concat(wf_returns_list)

ew_returns_wf = (daily_returns @ equal_weight).reindex(wf_returns.index)
spy_returns_wf = benchmark_returns.squeeze().reindex(wf_returns.index)

wf_cum = (1 + wf_returns).cumprod()
ew_cum_wf = (1 + ew_returns_wf).cumprod()
spy_cum_wf = (1 + spy_returns_wf).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(wf_cum, label='Walk-Forward Portfolio')
plt.plot(ew_cum_wf, label='Equal-Weight')
plt.plot(spy_cum_wf, label='SPY')
plt.title('Walk-Forward Backtest: Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

performance_metrics(wf_returns, 'Walk-Forward Portfolio')
# ✅ WALK-FORWARD BACKTEST complete


# === LEDOIT-WOLF SHRINKAGE ===
# Shrinkage reduces estimation noise even for N=8; a short historical window still produces noisy off-diagonal entries
lw = LedoitWolf()
lw.fit(daily_returns)
sigma_lw = lw.covariance_ * 252
print(f"Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")

# Tangency portfolio: maximizes Sharpe ratio via variable substitution
mu_clipped_lw = np.maximum(mu, rf + 0.001)
y_lw = cp.Variable(n)
k_lw = cp.Variable()
prob_lw = cp.Problem(
    cp.Minimize(cp.quad_form(y_lw, sigma_lw)),
    [(mu_clipped_lw - rf) @ y_lw == 1, cp.sum(y_lw) == k_lw,
     y_lw >= 0, y_lw <= 0.20 * k_lw, k_lw >= 0]
)
prob_lw.solve()
if y_lw.value is not None and k_lw.value is not None and k_lw.value > 1e-8:
    optimal_weights_lw = y_lw.value / k_lw.value
else:
    print("WARNING: LW tangency optimization failed")
    optimal_weights_lw = np.ones(n) / n

portfolio_returns_lw = daily_returns @ optimal_weights_lw
performance_metrics(portfolio_returns_lw, "LW Portfolio")
# ✅ LEDOIT-WOLF SHRINKAGE complete


# === BLACK-LITTERMAN + LEDOIT-WOLF COMBINED PIPELINE ===

# Step 1: equal-weight proxy for market portfolio
# TODO: replace w_eq with real market-cap weights if ETF AUM data becomes available
w_eq = np.ones(n) / n

# Step 2: reverse optimization for equilibrium returns
delta_ra = 2.5
Pi = delta_ra * sigma_lw @ w_eq

# Step 3: views
# Views are absolute expected returns, not relative — avoids equilibrium misinterpretation
P = np.zeros((2, n))
P[0, 5] = 1    # GLD: absolute return view (index 5)
P[1, 3] = 1    # TLT: absolute return view (index 3)
Q = np.array([0.08, 0.02])         # GLD: 8% annually, TLT: 2% annually
Omega = np.diag([0.0001, 0.0001])  # high confidence in both views

# Step 4: Bayesian update for BL expected returns
tau = 0.05
M = np.linalg.inv(tau * sigma_lw) + P.T @ np.linalg.inv(Omega) @ P
mu_bl = np.linalg.inv(M) @ (np.linalg.inv(tau * sigma_lw) @ Pi + P.T @ np.linalg.inv(Omega) @ Q)

# Step 5: Tangency portfolio: maximizes Sharpe ratio via variable substitution
mu_bl_clipped = np.maximum(mu_bl, rf + 0.001)
y_bl = cp.Variable(n)
k_bl = cp.Variable()
prob_bl = cp.Problem(
    cp.Minimize(cp.quad_form(y_bl, sigma_lw)),
    [(mu_bl_clipped - rf) @ y_bl == 1, cp.sum(y_bl) == k_bl,
     y_bl >= 0, y_bl <= 0.20 * k_bl, k_bl >= 0]
)
prob_bl.solve()
if y_bl.value is not None and k_bl.value is not None and k_bl.value > 1e-8:
    optimal_weights_bl = y_bl.value / k_bl.value
else:
    print("WARNING: BL+LW tangency optimization failed")
    optimal_weights_bl = np.ones(n) / n

# Step 6: weight comparison table
weights_df = pd.DataFrame({
    'MVO': optimal_weights,
    'LW': optimal_weights_lw,
    'BL+LW': optimal_weights_bl
}, index=tickers)
print("\nWeight Comparison: MVO vs LW vs BL+LW")
print(weights_df.round(4))

# Step 7: backtest BL+LW on full daily returns
portfolio_returns_bl = daily_returns @ optimal_weights_bl
performance_metrics(portfolio_returns_bl, "BL + LW Portfolio")
# ✅ BLACK-LITTERMAN + LEDOIT-WOLF COMBINED PIPELINE complete


# === PORTFOLIO PERFORMANCE SUMMARY ===
def _metrics_dict(returns):
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    ann_return = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = (ann_return - rf) / ann_vol
    max_dd = float(((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min())
    return {
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
    }

print("\n=== PORTFOLIO PERFORMANCE SUMMARY ===\n")
summary = pd.DataFrame({
    'Original MVO':        _metrics_dict(portfolio_daily_returns),
    'Walk-Forward BL+LW':  _metrics_dict(wf_returns),
    'LW Only':             _metrics_dict(portfolio_returns_lw),
    'BL + LW':             _metrics_dict(portfolio_returns_bl),
})
print(summary.round(4))
# ✅ PORTFOLIO PERFORMANCE SUMMARY complete
