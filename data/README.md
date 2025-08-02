# Data Directory 

This directory contains the raw and processed data used in the Multi-Asset Portfolio Optimizer project.

## Contents

| File | Description |
| ---- | ----------- |
| `multiasset_benchmark.csv` | Contains cumulative returns of the SPY benchmark for comparison. |
| `multiasset_closing_prices.csv` | Daily closing prices for all selected multi-asset ETFs. |
| `multiasset_cum_returns` | Cumulative returns of the optimized portfolio over the analysis period. |
| `multiasset_daily_returns` | Daily log returns calculated from closing prices for all assets. |
| `multiasset_ewc.csv` | Cumulative returns of the equal-weighted portfolio for performance benchmarking. |
| `multiasset_stats.csv` | Summary statistics including mean daily returns, standard deviations, and annualized metrics. |
| `efficient_frontier.csv` | Volatility and expected returns of optimized portfolios along the efficient frontier. |
| `optimal_weights.csv` | Final asset allocations for the optimized portfolio under defined constraints. |
| `performance_metrics.csv` | Key performance indicators including return, volatility, Sharpe ratio, and max drawdown. |  

All data was obtained directly through `yfinance` API calls. 

## Notes

- All `.csv` files are generated from the pipeline defined in the `notebooks/Final_Model/` notebook.
- Files are updated everytime the pipeline is run.

## Related Information

- Plots generated from this data are stored in the `data/` directory.
- Code used to generate this data is available in the `notebooks/` directory.
