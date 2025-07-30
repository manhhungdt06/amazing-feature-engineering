# Generated from: day_4.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import talib as ta
import numpy as np

# pd.options.display.float_format = '{:.1f}'.format
pd.options.display.float_format = '{:.2f}'.format


import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from sys import path

notebook_path = Path.cwd()
SITE = notebook_path.parent
path.append(str(SITE.absolute()))
from libs.a_helpers import *
from libs.c_helpers import *


ticker = "VNI"
filename = SITE / f'investing/data/exports/indices/{ticker}.csv'
df = pd.read_csv(filename, infer_datetime_format=True)[:-1]
df['Volume'] = df['Volume'].astype(int)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)


df.rename(columns={'Volume': 'Vol', 'Change': 'Chg'}, inplace=True)


# daily_df = df.copy()
# # daily_df['Return'] = 1 + (daily_df['Change'] / 100)
# # daily_df['CumReturn'] = daily_df['Return'].cumprod()


df.tail()


weekly_df = df.resample('W').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Vol': 'sum', 'Chg': 'sum'})

weekly_df['WLowMax'] = df.groupby(pd.Grouper(freq='W'))['Low'].max()
weekly_df['WHighMin'] = df.groupby(pd.Grouper(freq='w'))['High'].min()
weekly_df['WM_Diff'] = weekly_df['WLowMax'] - weekly_df['WHighMin']

# weekly_df['WM_Diff'] = abs(weekly_df['WLowMax'] - weekly_df['WHighMin'])
# weekly_df['WM_Stat'] = weekly_df.apply(lambda row: 1 if row['WLowMax'] < row['WHighMin'] else 0, axis=1)

weekly_df['WStat'] = weekly_df.apply(lambda row: 1 if row['Chg'] > 0 else 0, axis=1)

# weekly_df.to_csv(f'data/{filename}_OK_Weekly.csv')


weekly_df.tail(10)


monthly_df = weekly_df.resample('M').agg({'Close': 'last','Open': 'first','High': 'max','Low': 'min','Vol': 'sum','Chg': 'sum'})

# monthly_df['Month'] = monthly_df.index.strftime('%b')

monthly_df['MLowMax'] = weekly_df.groupby(pd.Grouper(freq='M'))['Low'].max()
monthly_df['MHighMin'] = weekly_df.groupby(pd.Grouper(freq='M'))['High'].min()

monthly_df['MM_Diff'] = monthly_df['MLowMax'] - monthly_df['MHighMin']
monthly_df['MStat'] = monthly_df.apply(lambda row: 1 if row['Chg'] > 0 else 0, axis=1)

# monthly_df.to_csv(f'data/{filename}_OK_Monthly.csv')


monthly_df.tail(10)


yearly_df = monthly_df.resample('Y').agg({'Close': 'last','Open': 'first','High': 'max','Low': 'min','Vol': 'sum','Chg': 'sum'})

yearly_df['YLowMax'] = weekly_df.groupby(pd.Grouper(freq='Y'))['Low'].max()
yearly_df['YHighMin'] = weekly_df.groupby(pd.Grouper(freq='Y'))['High'].min()

yearly_df['YM_Diff'] = yearly_df['YLowMax'] - yearly_df['YHighMin']
yearly_df['YStat'] = yearly_df.apply(lambda row: 1 if row['Chg'] > 0 else 0, axis=1)


yearly_df


# three_yearly_df = df.resample('3Y').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Volume': 'sum', 'Change': 'sum'})
# three_yearly_df

