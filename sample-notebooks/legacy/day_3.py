# Generated from: day_3.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Part 1


import pandas as pd
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 8, 6

import seaborn as sns
sns.set()

pd.options.display.float_format = '{:.5f}'.format


import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from sys import path

notebook_path = Path.cwd()
SITE = notebook_path.parent
path.append(str(SITE.absolute()))
from libs.helpers import *


ticker = "VNI"
filename = SITE / f'investing/data/exports/indices/{ticker}.csv'
df = pd.read_csv(filename, infer_datetime_format=True)[:-1]
df['Volume'] = df['Volume'].astype(int)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)


df_close = df['Close']
df_returns = np.log(df_close).diff()
df_returns.dropna(inplace=True)
len(df_returns)


df_returns[-252:].plot(title=f'{ticker} Returns')


df_returns.describe()


from scipy import stats


n , minmax, mean, var, skew, kurt = stats.describe(df_returns)
mini, maxi = minmax
std = var ** 0.5


plt.hist(df_returns, bins=15)


mean, std, n


from scipy.stats import norm
x = norm.rvs(mean, std, n)  # generate random numbers from a normal distribution


plt.hist(x, bins=15)


x_test = stats.kurtosistest(x)
x_test


df_test = stats.kurtosistest(df_returns)
# df_test = stats.kurtosistest(df_returns[-252:])
df_test


print(f'{"     Test statistic":20}{"p-value":>15}')
print(f'{" "*5}{"-"* 30}')
print(f"x:{x_test[0]:>17.2f}{x_test[1]:16.4f}")
print(f"VNI: {df_test[0]:13.2f}{df_test[1]:16.4f}")


plt.hist(df_returns, bins = 25, edgecolor='w', density= True)
# plt.hist(df_returns[-252:], bins = 25, edgecolor='w', density= True)
data = np.linspace(mini, maxi, 100)
plt.plot(data, norm.pdf(data, mean, std))


plt.hist(x, bins =25, density = True)
b = np.linspace(mini, maxi, 100)
plt.plot(b,stats.norm.pdf(b, mean, std))


stats.ttest_1samp(df_returns, 0, alternative='two-sided')


stats.ttest_1samp(df_returns.sample(252), 0, alternative='two-sided')



# # Part 2


df_close = pd.DataFrame(df_close, columns=['Close'])
df_close['lag1'] = df_close.Close.shift(1)
df_close['lag2'] = df_close.Close.shift(2)
df_close.dropna(inplace = True)
df_close.head()


lr = np.linalg.lstsq(df_close[['lag1', 'lag2']], df_close.Close, rcond=None)[0]


df_close['predict'] = np.dot(df_close[['lag1', 'lag2']], lr)
df_close.head()


# df_close.iloc[-252:][['Close', 'predict']].plot()
df_close.iloc[-52:][['Close', 'predict']].plot()

