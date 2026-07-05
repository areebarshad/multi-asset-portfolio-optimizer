import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === DATA ACQUISITION ===

#define multi-asset tickers (15-asset expanded universe)
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

# === RETURN CALCULATION ===

#calculate daily log returns
daily_returns = np.log(data / data.shift(1)).dropna()

#calculate monthly log returns (month-end resampling)
monthly_prices = data.resample('ME').last()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

# === EXPLORATORY DATA ANALYSIS ===

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
