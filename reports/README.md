# Portfolio Optimization Final Report

This report summarizes the **key findings** and **analytical results** from the Milti-Asset Portfolio Optimization Model.

## 1. Objective

To build an optimized multi-asset portfolio that:

- Minimizes portfolio **volatility**
- Achieves a **target annual return** of 5%
- Respects the following constraints:
  - Fully invested portfolio
  - No short-selling
  - Max 20% allocation per asset
 
## 2. Data Summary

Values derived from mean daily returns scaled to annual using 252 trading days. 

| Ticker | Annualized Return | Annualized Volatility |
| ------ | ----------------- | --------------------- |
| `DBC` | 2.99% | 17.90% |
| `EWJ` | 6.53% | 17.45% |
| `GLD` | 9.50% | 14.46% |
| `LQD` | 2.50% | 8.58% |
| `TLT` | -0.94% | 15.25% |
| `VEA` | 6.92% | 17.54% |
| `VNQ` | 4.74% | 20.99% |
| `VWO` | 5.02% | 19.94% |

## 3. Optimization Results

| Metric | Value |
| ------ | ----- |
| **Expected Portfolio Return** | 5% |
| **Expected Portfolio Volatility** | 9.19% |

## 4. Portfolio Weights

### Robust Optimal Weights

| Ticker | Weight |
| ------ | ------ |
| `DBC` | 20.00% |
| `EWJ` | 20.00% |
| `GLD` | 20.00% |
| `LQD` | 13.22% |
| `TLT` | 15.69% |
| `VEA` | 0.00% |
| `VNQ` | 0.00% |
| `VWO` | 0.00% |

- Heavy allocation to commodities, Japan equities, and gold
- No allocation to VEA, VNQ, VWO under these constraints

### Optimal Weights with Turnover Constraint (10%)

| Ticker | Weight |
| ------ | ------ |
| `DBC` | 12.00% |
| `EWJ` | 13.00% |
| `GLD` | 15.00% |
| `LQD` | 9.30% |
| `TLT` | 20.00% |
| `VEA` | 10.70% |
| `VNQ` | 10.00% |
| `VWO` | 10.00% |

- Diversifies across all assets while limiting weight changes for practical rebalancing

---

## 5. Performance Metrics 

| Portfolio | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown |
| --------- | ------------- | ----------------- | ------------ | ------------ |
| **Optimized** | 4.72% | 11.20% | 0.42 | -25.29% |
| **Equal-Weight** | 4.66% | 10.53% | 0.44 | -23.27% |
| **SPY Benchmark** | 12.22% | 18.10% | 0.68 | -35.75% |

### Insights

- The **Optimized** portfolio achieved target return with lower volatility than the **SPY Benchmark**
- The sharpe ratio for the **Optimized** portfolio was slight less than the **Equal-Weight** portfolio, indicating trade-offs in optimization constraints
- The annual voltility and max drawdowns were both significantly higher than the annual returns for the **SPY Benchmark** portfolio

## Conclusion

This portfolio optimization model demonstrates:

- The power of mean-variance optimization in balancing risk and return
- The impact of constraints on achievable risk-return profiles
- The value of backtesting to evaluate practical performance VS theoretical optimality 
