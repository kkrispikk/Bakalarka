# Bakalarka

import pandas as pd
from matplotlib import pyplot
import numpy as np

# Prices 

fname_prices = "/Users/kristian/bakalarka/MonthlyPrices.csv"
data = pd.read_csv(fname_prices, ";")

data["date"] = pd.to_datetime(data["date"], format = "%YM%m")
prices = pd.DataFrame(data)
prices = prices.set_index("date")
prices.index.name = "Month"
prices = prices[["Wheat, US HRW"]]
prices = prices.loc["1960-02-01":]
print(prices_short.head())

# SPEI (change positives to negatives)

fname_spei = "/Users/kristian/Desktop/bakalářka/data sucho/wheat (US great plains)  - 160.csv"
spei = pd.read_csv(fname_spei, ";")
spei["DATA"] = pd.to_datetime(spei["DATA"], format = "%b%Y")
spei_df = pd.DataFrame(spei)
spei_df = spei_df.set_index("DATA")
spei1 = spei_df["SPEI_3"]
spei1 = spei1.loc["1960-02-01":]
spei2 = np.array([-x for x in spei1])
spei2 = pd.DataFrame({"SPEI_3": spei2}, index = prices_short.index)

print(spei2.head())
#np.corrcoef(prices,spei2)
pyplot.plot(spei2)
pyplot.plot(prices_short)


# Relative changes
## Prices 

relative_changes = prices_short.pct_change()
relative_changes = relative_changes.dropna()
relative_changes = pd.DataFrame(relative_changes)
relative_changes = relative_changes *100
pyplot.plot(relative_changes)
print(relative_changes)

# Dickey-Fuller test for non-staionarity:prices
from statsmodels.tsa.stattools import adfuller
X = relative_changes.iloc[:,0].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
    
# Dickey-Fuller test for non-staionarity: spei
X = spei2.iloc[:,0].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))                   

# OLS before filtering

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

ols = pd.concat([relative_changes, spei2], axis = 1)
ols = ols["1960-03-01":]
print(ols)
Y = ols["Wheat, US HRW"] 
X = ols["SPEI_3"]

model = sm.OLS(Y,X)
results = model.fit()
print(results.params)
print(results.pvalues)
rmse = np.sqrt(mean_squared_error(Y,X))
r2 = r2_score(Y,X)
print(rmse)
print(r2)
print(np.corrcoef(X,Y))

# Filter
j = 5

def movingaverage_shorter (values, window):
    weights = np.repeat(1.0, window)/(window)
    sma = np.convolve(values, weights, 'valid')
    return sma

def movingaverage_longer (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

priceMA = movingaverage_shorter(X[2**(j):],2**j) - movingaverage_longer(X,2**(j+1))
speiMA = movingaverage_shorter(Y[2**(j):],2**j) - movingaverage_longer(Y,2**(j+1))

speiMA1 = pd.DataFrame({"Rice, Thai 5%": speiMA}, index = relative_changes.index[2**(j+1)-1:])
priceMA1 = pd.DataFrame({"SPEI_3": priceMA}, index = relative_changes.index[2**(j+1)-1:])
ols_filter = pd.concat([priceMA1, speiMA1], axis = 1)
ols_filter

pyplot.plot(priceMA1)
pyplot.plot(speiMA1)
np.corrcoef(priceMA,speiMA)

# OLS after filter

YY = ols_filter["Wheat, US HRW"]
XX = ols_filter["SPEI_3"]

model = sm.OLS(YY,XX)
results = model.fit()
results.summary(
