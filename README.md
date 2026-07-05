# Multi-Asset Portfolio Optimizer

## Project Overview

This project implements a multi-asset portfolio optimization model in Python using historical financial data. It builds an efficient, diversified portfolio by applying **tangency (max Sharpe) optimization** with **Ledoit-Wolf covariance shrinkage** and **Black-Litterman expected returns**, and evaluates performance through walk-forward backtesting and risk metrics.

---

## Objectives

- Download and clean historical price data for a diversified 15-ETF universe covering equities, fixed income, commodities, and real assets.
- Calculate daily and monthly log returns; visualize price trends, return distributions, and correlations.
- Optimize portfolio weights to maximize the Sharpe ratio (tangency portfolio), subject to constraints:
  - **Fully invested portfolio** (weights sum to 1)
  - **No short selling** (weights ≥ 0)
  - **Max 20% allocation per asset**
- Generate the efficient frontier with gross and net-of-transaction-cost curves, Capital Market Line, and tangency point.
- Backtest the optimized portfolio against:
  - An **equal-weight portfolio**
  - The **SPY benchmark (S&P 500)**
- Evaluate performance metrics:
  - **Annualized Return**
  - **Annualized Volatility**
  - **Sharpe Ratio** (risk-free rate = 4%)
  - **Maximum Drawdown**
- Extend the model with:
  - **Turnover constraints** for practical rebalancing limits
  - **Walk-forward backtesting** to measure out-of-sample performance
  - **Ledoit-Wolf covariance shrinkage** for stable estimation with 15 assets
  - **Black-Litterman + Ledoit-Wolf pipeline** for Bayesian expected return estimation

---

## Technology Used

**Python**:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `cvxpy`
- `scikit-learn` (Ledoit-Wolf)

---

## Asset Universe

15 ETFs spanning major asset classes (2015–2025):

| Ticker | Asset Class |
| ------ | ----------- |
| `VEA` | Developed Markets ex-US |
| `VWO` | Emerging Markets |
| `EWJ` | Japan Equities |
| `TLT` | 20+ Year US Treasury Bonds |
| `LQD` | Investment Grade Corporate Bonds |
| `GLD` | Gold |
| `VNQ` | REITs |
| `DBC` | Broad Commodities Index |
| `SPY` | US Large Cap (S&P 500) |
| `QQQ` | US Large Cap Tech (NASDAQ-100) |
| `IEF` | 7-10 Year US Treasury Bonds |
| `HYG` | High Yield Corporate Bonds |
| `SLV` | Silver |
| `GSG` | Commodity Index |
| `EMLC` | Emerging Market Local Currency Bonds |

---

## Key Steps

1. **Data Acquisition**: Download daily closing prices from 2015–2025 using `yfinance`. No CSV files — all data is fetched live.
2. **Data Cleaning and Preprocessing**: Handle missing values with forward-fill/back-fill; calculate daily and monthly log returns.
3. **Exploratory Data Analysis**: Plot price trends, return distributions, and a pairwise correlation heatmap.
4. **Tangency Portfolio Optimization**: Maximize Sharpe ratio via CVXPY quadratic programming using a variable substitution formulation.
5. **Efficient Frontier**: Sweep target returns to trace the frontier; overlay net-of-cost curve (10bps), Capital Market Line, and tangency point.
6. **Backtesting**: Compare optimized portfolio vs equal-weight portfolio vs SPY benchmark.
7. **Extensions**:
   - Turnover-constrained rebalancing (10% limit)
   - Walk-forward backtest with 3-year rolling training windows
   - Ledoit-Wolf covariance shrinkage
   - Black-Litterman + Ledoit-Wolf combined pipeline (views: GLD 8%, TLT 2%)
8. **Performance Summary**: Comparative table across MVO, LW, BL+LW, and walk-forward strategies.

---

## Folder Structure

```plaintext
/
├── notebooks/
│   ├── DataAcquisition_and_Preprocessing/
│   │   └── DataAcquisition_Preprocessing.py
│   ├── Portfolio_Optimization/
│   │   └── Portfolio_Optimization.py
│   ├── Backtesting/
│   │   └── Backtesting.py
│   ├── Extensions_Dashboard/
│   │   └── Extensions_Dashboard.py
│   ├── Final_Model/
│   │   └── model.py
│   └── README.md
├── plots/
│   ├── Closing_Prices/
│   ├── Daily_Return_Distributions/
│   ├── Returns_Correlation_Heatmap/
│   ├── Optimal_Weights/
│   ├── Efficient_Frontier/
│   ├── Cumulative_Returns/
│   ├── Walk_Forward_Backtest/
│   └── README.md
├── reports/
│   └── README.md
└── README.md
```

---

## Why This Project Matters

Portfolio optimization sits at the intersection of financial mathematics, statistics, and convex optimization. This project demonstrates:

- **End-to-end data pipeline**: from live data acquisition to performance evaluation
- **Convex optimization**: CVXPY quadratic programming with real-world constraints
- **Bayesian estimation**: Black-Litterman model for incorporating analyst views into expected returns
- **Robust statistics**: Ledoit-Wolf shrinkage to handle estimation error in high-dimensional covariance matrices
- **Rigorous backtesting**: walk-forward evaluation to measure genuine out-of-sample performance
- **Risk-reward analysis**: efficient frontier, Sharpe ratio, max drawdown, and transaction cost modeling

---

## Author

**Areeb Arshad**

Sophomore, Data Science & Economics

Virginia Tech
