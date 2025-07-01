# Notebooks Directory

This directory contains the Google Colab notebooks for the Multi-Asset Portfolio Optimizer project.

## Notebook Breakdown

| Notebook | Description |
| -------- | ----------- |
| `Backtesting/` | Evaluates optimized portfolio performance against equal-weight and SPY benchmark using historical returns. |
| `DataAcquisition_and_Preprocessing/` | Downloads multi-asset financial data, cleans missing values, and calculats daily and monthly returns. |
| `Extensions_Dashboard/` | Contains exploratory extensions such as turnover-constrained optimization and visualization dashboard setups. |
| `Portfolio_Optimization/` | Implements mean-variance portfolio optimization with practical constraints to derive optimal asset allocations. |
| `Final_Model/` | Integrates data acquisition, optimization, backtesting, and evaluation into a single, final model. |

## Notes

- All data files required for the notebooks are located in the `data/` directory. The raw sector data is fetched via `yFinance`.
- The notebooks are executed in sequence, starting from data acquisition -> portfolio optimization -> backtesting -> extensions dashboard.

## Related Information

- The data folder contains the raw and processed data used in the notebooks.
- The plots generated during the model's run are saved in the `plots/` directory.
