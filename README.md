# Bakalarka
# here is the code for my bachelor thesis

import pandas as pd
from matplotlib import pyplot
import numpy as np
# prices wheat

fname_prices = "/Users/kristian/bakalarka/MonthlyPrices.csv"
data = pd.read_csv(fname_prices, ";")

data["date"] = pd.to_datetime(data["date"], format = "%YM%m")
df = pd.DataFrame(data)
df = df.set_index("date")
df.index.name = "Month"
df = df[["Wheat, US HRW"]]
df_short = df.loc["1960-02-01":]
print(df_short.head())

# SPEI (change positives to negatives)

fname_spei = "/Users/kristian/Desktop/bakalářka/data sucho/wheat (US great plains)  - 160.csv"
wheat = pd.read_csv(fname_spei, ";")
wheat["DATA"] = pd.to_datetime(wheat["DATA"], format = "%b%Y")
wheat_df = pd.DataFrame(wheat)
wheat_df = wheat_df.set_index("DATA")
wheat1 = wheat_df["SPEI_3"]
wheat1 = wheat1.loc["1960-02-01":]
wheat2 = np.array([-x for x in wheat1])
wheat2 = pd.DataFrame({"SPEI_3": wheat2}, index = df_short.index)

print(wheat2.head())
#np.corrcoef(df,wheat2)
pyplot.plot(wheat2)
pyplot.plot(df_short)


## relative changes
# prices 

relative_changes = df_short.pct_change()
relative_changes = relative_changes.dropna()
relative_changes = pd.DataFrame(relative_changes)
relative_changes = relative_changes *30
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
from statsmodels.tsa.stattools import adfuller
X = wheat2.iloc[:,0].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))                   

# OLS before filtering

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

wheat_ols = pd.concat([relative_changes, wheat2], axis = 1)
wheat_ols = wheat_ols["1960-03-01":]
print(wheat_ols)
Y = wheat_ols["Wheat, US HRW"] 
X = wheat_ols["SPEI_3"]

model = sm.OLS(Y,X)
results = model.fit()
print(results.params)
print(results.pvalues)
rmse = np.sqrt(mean_squared_error(Y,X))
r2 = r2_score(Y,X)
print(rmse)
print(r2)
print(np.corrcoef(X,Y))

# filter
j = 5

def movingaverage_shorter (values, window):
    weights = np.repeat(1.0, window)/2**(j-1)
    sma = np.convolve(values, weights, 'valid')
    return sma

def movingaverage_longer (values, window):
    weights = np.repeat(1.0, window)/2**(j)
    sma = np.convolve(values, weights, 'valid')
    return sma

priceMA = movingaverage_shorter(X[1:],j) - movingaverage_longer(X,j+1)
speiMA = (movingaverage_shorter(Y[1:],j) - movingaverage_longer(Y,j+1))

speiMA1 = pd.DataFrame({"Wheat, US HRW": speiMA}, index = relative_changes.index[j:])
priceMA1 = pd.DataFrame({"SPEI_3": priceMA}, index = relative_changes.index[j:])
wheat_ols = pd.concat([priceMA1, speiMA1], axis = 1)
wheat_ols

pyplot.plot(priceMA1)
pyplot.plot(speiMA1)
np.corrcoef(priceMA,speiMA)
