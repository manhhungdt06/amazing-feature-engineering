# Generated from: day_1.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import talib as ta
import numpy as np

pd.options.display.float_format = '{:.2f}'.format


import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from sys import path

notebook_path = Path.cwd()
SITE = notebook_path.parent
path.append(str(SITE.absolute()))
from libs.c_helpers import *


folder_path = Path(SITE / 'investing/data/exports/indices/')
filenames = [f for f in folder_path.iterdir() if f.is_file()]

cumulative_returns = {}

for filename in filenames:
    ticker = str(filename).split('.')[0].split('/')[-1]
    df = pd.read_csv(filename, infer_datetime_format=True)[:-1]
    
    df['Return'] = 1 + (df['Change'] / 100)
    df['CumReturn'] = df['Return'].cumprod()
    # print(ticker, list(df.tail(1)['CumReturn'])[0])
    cumulative_returns[ticker] = list(df.tail(1)['CumReturn'])[0]

sorted_returns = dict(sorted(cumulative_returns.items(), key=lambda x: x[1], reverse=True))
sorted_returns



ticker = "VNI"
filename = SITE / f'investing/data/exports/indices/{ticker}.csv'
df = pd.read_csv(filename, infer_datetime_format=True)[:-1]
df['Volume'] = df['Volume'].astype(int)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)


daily_df = df.copy()
daily_df['Return'] = 1 + (daily_df['Change'] / 100)
daily_df['CumReturn'] = daily_df['Return'].cumprod()
daily_df


weekly_df = df.resample('W').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Volume': 'sum', 'Change': 'sum'})
# weekly_df['AvgVol_1W'] = weekly_df['Volume'].rolling(1).mean()
# weekly_df['AvgVol_2W'] = weekly_df['Volume'].rolling(2).mean()
# weekly_df['AvgVol_5W'] = weekly_df['Volume'].rolling(5).mean()
weekly_df


monthly_df = df.resample('M').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Volume': 'sum', 'Change': 'sum'})
monthly_df


yearly_df = df.resample('Y').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Volume': 'sum', 'Change': 'sum'})
yearly_df


three_yearly_df = df.resample('3Y').agg({'Close': 'last', 'Open': 'first', 'High': 'max', 'Low': 'min', 'Volume': 'sum', 'Change': 'sum'})
three_yearly_df

