# Bakalarka
#here is the code for my bachelor thesis

import pandas as pd
from matplotlib import pyplot
import numpy as np
#prices wheat

fname_prices = "/Users/kristian/bakalarka/MonthlyPrices.csv"
data = pd.read_csv(fname_prices, ";")

data["date"] = pd.to_datetime(data["date"], format = "%YM%m")
df = pd.DataFrame(data)
df = df.set_index("date")
df.index.name = "Month"
df = df[["Wheat, US HRW"]]
df_short = df.loc["1960-02-01":]
print(df_short.head())

#SPEI Wheat (change positives to negatives)

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
