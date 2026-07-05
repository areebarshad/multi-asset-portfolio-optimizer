# Portfolio Optimization Final Report

This report summarizes the **key findings** and **analytical results** from the Multi-Asset Portfolio Optimization Model.

## 1. Objective

Build an optimized multi-asset portfolio that:

- Maximizes the **Sharpe ratio** (tangency portfolio formulation)
- Respects the following constraints:
  - Fully invested (weights sum to 1)
  - No short selling (weights ≥ 0)
  - Maximum 20% allocation per asset

## 2. Asset Universe

15-asset ETF universe covering equities, fixed income, commodities, and real assets (2015–2025):

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

## 3. Methodology

### Optimization: Tangency Portfolio (Max Sharpe)

The final model solves for the **tangency portfolio** — the point on the efficient frontier that maximizes the Sharpe ratio — using a variable substitution to convert the fractional-programming problem into a convex quadratic program solved by CVXPY.

Risk-free rate: **4%** (approximation for the 2015–2025 period).

### Covariance Estimation: Ledoit-Wolf Shrinkage

Sample covariance matrices are noisy with 15 assets. The model applies **Ledoit-Wolf shrinkage** to regularize the off-diagonal estimates and produce a more stable covariance matrix for optimization.

### Expected Returns: Black-Litterman + Ledoit-Wolf

The **Black-Litterman model** blends equilibrium returns (derived by reverse-optimizing from equal weights) with two analyst views:

- **GLD (Gold)**: 8% annual return
- **TLT (Long Bonds)**: 2% annual return

The Bayesian update produces posterior expected returns used in the BL+LW tangency portfolio.

### Walk-Forward Backtest

The model is evaluated out-of-sample using a **walk-forward backtest**: a 3-year rolling training window re-estimates Ledoit-Wolf covariances and BL+LW expected returns each period, with portfolio weights applied to the next year's returns. This avoids look-ahead bias.

### Transaction Costs

The **net efficient frontier** applies a 10bps cost per unit of turnover relative to the equal-weight baseline, allowing direct comparison of gross vs net risk-return profiles.

## 4. Key Changes from Original Model

| Dimension | Original | Final |
| --------- | -------- | ----- |
| Assets | 8 ETFs | 15 ETFs |
| Objective | Min variance at 5% target return | Max Sharpe (tangency portfolio) |
| Covariance | Sample covariance | Ledoit-Wolf shrinkage |
| Expected returns | Historical sample mean | BL + LW Bayesian update |
| Backtest | Single in-sample period | Walk-forward (3-year rolling window) |
| Transaction costs | Not modeled | 10bps net frontier |
| Sharpe calculation | No risk-free adjustment | rf = 4% subtracted |

## 5. Portfolio Performance Summary

Performance metrics are computed from running `notebooks/Final_Model/model.py`. The summary table compares four strategies:

| Strategy | Description |
| -------- | ----------- |
| **Original MVO** | Tangency portfolio using sample covariance and historical mean returns |
| **LW Only** | Tangency portfolio with Ledoit-Wolf covariance shrinkage |
| **BL + LW** | Tangency portfolio with BL expected returns and LW covariance (full pipeline) |
| **Walk-Forward BL+LW** | Out-of-sample walk-forward evaluation of the BL+LW strategy |

All strategies are benchmarked against an **equal-weight portfolio** and **SPY**.

## 6. Plots

Generated plots are stored in the `plots/` directory:

- Closing prices (all 15 assets)
- Daily return distributions
- Correlation heatmap
- Optimal portfolio weights (tangency)
- Efficient frontier (gross + net-of-costs + CML + tangency point)
- Cumulative returns vs equal-weight and SPY
- Walk-forward backtest cumulative returns

## Conclusion

The final model demonstrates:

- How expanding the asset universe and switching to a max-Sharpe objective changes portfolio composition
- The benefit of Ledoit-Wolf shrinkage for stable covariance estimation with 15 assets
- How Black-Litterman views can tilt a portfolio toward assets with specific return expectations
- The importance of walk-forward backtesting to measure true out-of-sample performance
- The impact of transaction costs on net realized returns along the efficient frontier
