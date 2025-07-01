# Multi-Asset Portfolio Optimizer

## Project Overview

- This project implements a multi-asset portfolio optimization model in Python using historical financial data.
- It builds an efficient, diversified portfolio by applying mean-variance optimization (**Markowitz Portfolio Theory**) under practical constraints and analyzes its performance through backtesting and risk metrics.

---

## Objectives

- Download and clean historical data for a diversified set of ETFs and asset classes.
- Calculate daily and monthly log returns, visualize trends, distributions, and correlations.
- Optimize portfolio weights to minimize risk for a target return, with constraints:
  - **Fully invested portfolio**
  - **No short selling**
  - **Max 20% allocation per asset**
- Generate the efficient frontier to analyze risk-return trade-offs.
- Backtest the optimized portfolio against:
  - **An equal-weight portfolio**
  - **An `SPY` benchmark (S&P 500)**
- Calculate performance metrics:
  - **Annualized Return**
  - **Annualized Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**
- Extend with **turnover constraints** to model practical rebalancing limits.

---

## Technology Used

**Python**:
- `Pandas`
- `NumPy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `cvxpy`

## Assets Used

- `VEA` -> Developed Markets ex-US
- `VWO` -> Emerging Markets
- `EWJ` -> Japan
- `TLT` -> 20+ Year Treasury Bonds
- `LQD` -> Investment Grade Corporate Bonds
- `GLD` -> Gold
- `VNQ` -> REITs
- `DBC` -> Commodities Index

---

## Key Steps

1. **Data Acquisition**: Download daily closing prices from 2015-2025 uding `yfinance`.
2. **Data Cleaning and Preprocessing**: Handle missing values, calculate daily and monthly log returns.
3. **Exploratory Data Analysis**: Plot -> Price Trends, Return Distributions, Correlation Heatmap.
4. **Mean-Variance Optimization**: Formulate the objective, set the constraints, solve using cvxpy quadratic programming.
5. **Efficient Frontier**: Calculate portfolios for varying target returns, visualize risk-return trade-offs.
6. **Backtesting**: Optimized portfolio VS equal-weight portfolio VS SPY benchmark
7. **Extensions**: Turnover constraints to limit drastic rebalancing.
8. **Display results**: View Reports folder.

---

## Future Extensions 

- Black-Litterman model for incorporating market views
- Robust optimization under parameter uncertainty
- Transaction cost modeling
- Interactive dashboard using Streamlit

---

## Why this Project is Important:

- Portfolio optimization is a core concept in quantitative finance and asset management.
- This project demonstrates the ability to:
  - **Integrate financial mathematics, statistics, and optimization**
  - **Build a full data pipeline from acquisition to analysis and evaluation**
  - **Apply convex optimization for real-world problems**
  - **Analyze risk-reward trade-offs for informed investment decisions**

---
 
## Author

**Areeb Arshad**

Sophomore, Data Science, Economics

Virginia Tech

---

## Folder Structure

```plaintext
/
├── data/
│   ├── multiasset_benchmark.csv             
│   ├── multiasset_closing_prices.csv            
│   ├── multiasset_cum_returns.csv
|   ├── multiasset_daily_returns.csv
|   ├── multiasset_ewc.csv
|   ├── multiasset_stats.csv
|   ├── README.md                       
├── notebooks/
│   ├── Backtesting/      
│   ├── DataAcquisition_and_Preprocessing/            
│   ├── Extensions_Dashboard/
│   ├── Final_Model/
|   ├── Portfolio_Optimization/
|   ├── README.md     
├── plots/
│   ├── Closing_Prices/               
│   ├── Cumulative_Returns/ 
│   ├── Daily_Return_Distributions/       
│   ├── Efficient_Frontier/                     
│   ├── Optimal_Weights/               
│   ├── Returns_Correlation_Heatmap/
|   ├── README.md           
├── reports/
|   ├── README.md            
└── README.md
