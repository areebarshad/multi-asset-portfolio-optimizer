# Notebooks Directory

This directory contains the Python scripts for the Multi-Asset Portfolio Optimizer project. All files are `.py` and can be edited and run directly in VS Code.

## Script Breakdown

| Script | Description |
| ------ | ----------- |
| `DataAcquisition_and_Preprocessing/DataAcquisition_Preprocessing.py` | Downloads 15-asset ETF closing prices via `yfinance`, handles missing values, computes daily and monthly log returns, and produces EDA plots (price trends, return distributions, correlation heatmap). |
| `Portfolio_Optimization/Portfolio_Optimization.py` | Implements the tangency (max Sharpe) portfolio via CVXPY quadratic programming with constraints (fully invested, no short selling, ≤20% per asset). Plots portfolio weights and the efficient frontier with gross/net curves, Capital Market Line, and tangency point. |
| `Backtesting/Backtesting.py` | Backtests the optimized portfolio against an equal-weight portfolio and the SPY benchmark. Computes annualized return, volatility, Sharpe ratio, and max drawdown for each. |
| `Extensions_Dashboard/Extensions_Dashboard.py` | Extends the model with: turnover-constrained rebalancing, walk-forward backtesting using BL+LW weights, Ledoit-Wolf covariance shrinkage, and the full Black-Litterman + Ledoit-Wolf pipeline. Ends with a comparative performance summary table. |
| `Final_Model/model.py` | Single consolidated script that runs the entire pipeline end-to-end, from data acquisition through the BL+LW performance summary. |

## Execution Order

Each script re-downloads data independently and can be run standalone. The recommended sequence if running in segments is:

1. `DataAcquisition_and_Preprocessing`
2. `Portfolio_Optimization`
3. `Backtesting`
4. `Extensions_Dashboard`

## Notes

- All data is fetched live via `yfinance` — no CSV files are required.
- The plots generated during each script's run match those saved in the `plots/` directory.
- The final model expanded the asset universe from 8 to 15 ETFs and replaced min-variance optimization with a tangency (max Sharpe) formulation incorporating Ledoit-Wolf shrinkage and Black-Litterman views.
