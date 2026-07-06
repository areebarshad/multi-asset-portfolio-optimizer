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

Metrics generated from `notebooks/Final_Model/model.py` on **2026-07-06** (data: 2015-01-01 to 2025-12-31, 2,765 trading days, rf = 4%).

### Strategy Comparison

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
| -------- | ----------- | --------------- | ------------ | ------------ |
| **Original MVO** | **9.87%** | **10.55%** | **0.5562** | **-24.17%** |
| **LW Only** | 9.89% | 10.57% | 0.5570 | -24.16% |
| **BL + LW** | 8.16% | 14.43% | 0.2882 | -30.43% |
| **Walk-Forward BL+LW** | 7.39% | 12.25% | 0.2766 | -26.95% |
| *Equal-Weight (baseline)* | *6.26%* | *10.38%* | *0.2176* | *-23.76%* |
| *SPY Benchmark* | *12.70%* | *17.83%* | *0.4879* | *-35.75%* |

**Best Sharpe ratio**: LW Only (0.5570) — marginally beats MVO (0.5562).  
**Lowest drawdown**: MVO and LW Only both at ~-24.2%, vs SPY at -35.7%.  
**Ledoit-Wolf shrinkage coefficient**: 0.0113 (low shrinkage — sample covariance already well-conditioned with 10 years of data).

### Tangency Portfolio (MVO) Weights

| Ticker | Weight | Asset Class |
| ------ | ------ | ----------- |
| GLD    | 20.00% | Gold |
| IEF    | 20.00% | 7-10yr US Treasuries |
| QQQ    | 20.00% | NASDAQ-100 |
| SPY    | 20.00% | S&P 500 |
| SLV    | 10.35% | Silver |
| TLT    |  9.65% | 20+yr US Treasuries |

The model concentrates in gold (diversifier), short-to-mid duration bonds (volatility dampeners), and US equity growth factors — consistent with a max-Sharpe objective over the 2015–2025 bull market.

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
