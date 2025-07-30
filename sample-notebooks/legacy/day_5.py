# Generated from: day_5.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import talib as ta
import numpy as np
import json
import os

pd.options.display.max_columns = 34
pd.options.display.float_format = '{:.3f}'.format


import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
from sys import path

notebook_path = Path.cwd()
SITE = notebook_path.parent
path.append(str(SITE.absolute()))
from libs.helpers import *


current_dir = os.getcwd()
current_dir



file_path = os.path.join(current_dir, '../../data/investing/currencies/results/DXY.json')

with open(file_path, 'r') as file:
    data = json.load(file)

dxy = pd.DataFrame(data)
dxy = dxy[['day_', 'close_', 'open_', 'high_', 'low_']]
dxy.rename(columns={'day_': 'date', 'close_': 'c', 'open_': 'o', 'high_': 'h', 'low_': 'l'}, inplace=True)
dxy.set_index('date', inplace=True)
dxy.index = pd.to_datetime(dxy.index)
# dxy.index
for column in dxy.columns:
    dxy[column] = pd.to_numeric(dxy[column], errors='coerce')

dxy = dxy[::-1]
dxy['chg'] = dxy['c'].pct_change() * 100
# dxy['chg'] = (dxy['c'] - dxy['c'].shift(1)) * 100 / dxy['c'].shift(1)
# dxy.info()
dxy.dropna(inplace=True)
# dxy


# dxy = calculate_close_to_close_volatility(dxy, 'c', [5,], [21,])
# dxy = calculate_parkinson_volatility(dxy, 'h', 'l', [5,], [21,])
# dxy = calculate_garman_klass_volatility(dxy, 'h', 'l', 'c', 'o', [5,], [21,])
# dxy = calculate_hodges_tompkins_volatility(dxy, 'c', [5,], [21,])
# dxy = calculate_rogers_satchell_volatility(dxy, 'h', 'l', 'c', 'o', [5,], [21,])
# dxy = calculate_yang_zhang_volatility(dxy, 'h', 'l', 'c', 'o', [5,], [21,])


# functions = ['mean', 'var', 'std', 'skew', 'kurt']
# dxy = add_rolling_functions(dxy, ['chg',], ['5D', '21D'], functions)


# # dxy = add_percentage_change(dxy, 'c', ['W', 'M', '3M', 'YTD', 'Y'])
# dxy = add_percentage_change(dxy, 'c', ['W', 'M'])


# indicators = {
#     'EMA': {
#         'time_periods': [5, 8, 13, 21, 34],
#         'input_columns': 'c'
#     },
#     'TRANGE': {
#         'time_periods': "",
#         'input_columns': ('h', 'l', 'c')
#     }
# }
# dxy = add_technical_indicators(dxy, indicators)


# dxy.tail()


weekly_dxy = dxy.resample('W').agg({'c': 'last', 'o': 'first', 'h': 'max', 'l': 'min', 'chg': 'sum'})

weekly_dxy['WLowMax'] = dxy.groupby(pd.Grouper(freq='W'))['l'].max()
weekly_dxy['WHighMin'] = dxy.groupby(pd.Grouper(freq='w'))['h'].min()
weekly_dxy['WM_Diff'] = weekly_dxy['WLowMax'] - weekly_dxy['WHighMin']


weekly_dxy['WStat'] = weekly_dxy.apply(lambda row: 1 if row['chg'] > 0 else 0, axis=1)
weekly_dxy.tail(10)


monthly_dxy = weekly_dxy.resample('M').agg({'c': 'last','o': 'first','h': 'max','l': 'min', 'chg': 'sum'})

# monthly_dxy['Month'] = monthly_dxy.index.strftime('%b')

monthly_dxy['MLowMax'] = weekly_dxy.groupby(pd.Grouper(freq='M'))['l'].max()
monthly_dxy['MHighMin'] = weekly_dxy.groupby(pd.Grouper(freq='M'))['h'].min()

monthly_dxy['MM_Diff'] = monthly_dxy['MLowMax'] - monthly_dxy['MHighMin']
monthly_dxy['MStat'] = monthly_dxy.apply(lambda row: 1 if row['chg'] > 0 else 0, axis=1)
monthly_dxy.tail(10)

# monthly_dxy.to_csv(f'data/{filename}_OK_Monthly.csv')

