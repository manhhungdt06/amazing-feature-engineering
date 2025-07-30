# Generated from: day_2.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import talib as ta  # pip install numpy==1.23.0 --force 
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
from libs.helpers import *
# from libs.c_helpers import *


ticker = "VNI"
filename = SITE / f'investing/data/exports/indices/{ticker}.csv'

# ignore 2024-04-01 lol :))
df = pd.read_csv(filename, infer_datetime_format=True)[:-1]
df = df.rename(columns={
    # 'Date': 'D',
    # 'Close': 'C',
    # 'Open': 'O',
    # 'High': 'H',
    # 'Low': 'L',
    'Volume': 'Vol',
    'Change': 'Chg'
})


# df = pd.read_csv(filename, infer_datetime_format=True)
# df['Ret'] = 1 + (df['Chg'] / 100)
# df['CumRet'] = df['Ret'].cumprod()

df['Vol'] = df['Vol'].astype(int)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)


daily_df = df.copy()


daily_df.describe()


# daily_df['VolChg']  = daily_df['Vol'].pct_change()*100

# # daily_df['AVGPRICE'] = ta.AVGPRICE(daily_df['Open'], daily_df['High'], daily_df['Low'], daily_df['Close'])
# # daily_df['AVGPChg'] = daily_df['AVGPRICE'].pct_change() * 100

# daily_df['AVGPChg'] = (ta.AVGPRICE(daily_df['Open'], daily_df['High'], daily_df['Low'], daily_df['Close'])).pct_change() * 100

# # daily_df['MEDPRICE'] = ta.MEDPRICE(daily_df['High'], daily_df['Low'])
# # daily_df['MEDPChg'] = daily_df['MEDPRICE'].pct_change() * 100

# daily_df['MEDPChg'] = (ta.MEDPRICE(daily_df['High'], daily_df['Low'])).pct_change() * 100

# # daily_df['TYPPRICE'] = ta.TYPPRICE(daily_df['High'], daily_df['Low'], daily_df['Close'])
# # daily_df['TYPPChg'] = daily_df['TYPPRICE'].pct_change() * 100

# daily_df['TYPPChg'] = (ta.TYPPRICE(daily_df['High'], daily_df['Low'], daily_df['Close'])).pct_change() * 100

# # daily_df['WCLPRICE'] = ta.WCLPRICE(daily_df['High'], daily_df['Low'], daily_df['Close'])
# # daily_df['WCLPChg'] = daily_df['WCLPRICE'].pct_change() * 100

# daily_df['WCLPChg'] = (ta.WCLPRICE(daily_df['High'], daily_df['Low'], daily_df['Close'])).pct_change() * 100


# daily_df['Chg1W'] = daily_df['Close'].pct_change(5)*100
# daily_df['Chg1M'] = daily_df['Close'].pct_change(21)*100
# daily_df['YTD'] = daily_df['Close'] / daily_df.loc[daily_df.index[0], 'Close'] - 1
# daily_df['Chg1Y'] = daily_df['Close'].pct_change(252)*100
# daily_df['Chg3Y'] = daily_df['Close'].pct_change(252 * 3)*100


# period = 13
# daily_df['MFI'] = ta.MFI(daily_df['High'], daily_df['Low'], daily_df['Close'], daily_df['Volume'], timeperiod=period)
# daily_df['RSI'] = ta.RSI(daily_df['Close'], timeperiod=period)
# daily_df['ATR'] = ta.ATR(daily_df['High'], daily_df['Low'], daily_df['Close'], timeperiod=period)
# daily_df['NATR'] = (daily_df['ATR'] / daily_df['Close']) * 100 
# daily_df['TRANGE'] = ta.TRANGE(daily_df['High'], daily_df['Low'], daily_df['Close'])


daily_df2 = df.copy()
daily_df2


functions = ['mean',]
daily_df2 = add_rolling_functions(daily_df2, ['Chg',], ['8D',], functions)


daily_df2 = add_percentage_change(daily_df2, 'Close', ['W', 'M', 'YTD',])


indicators = {
    'EMA': {
        'time_periods': [5, 13],
        'input_columns': 'Close'
    },
    'TRANGE': {
        'time_periods': "",
        'input_columns': ('High', 'Low', 'Close')
    }
}


daily_df2 = add_technical_indicators(daily_df2, indicators)


daily_df2["VpR"] = daily_df2["Vol"] / daily_df2["TRANGE"]


daily_df2.tail()


daily_df2.info()


# daily_df2.loc[daily_df2['TRANGE'] == 0, 'VpR'] = 0


mask = daily_df2.isnull().any(axis=1)
rows_with_null = daily_df2[mask]
rows_with_null


df_non_null = daily_df2.dropna()
df_non_null


# columns_to_check = ['ChgW', 'ChgM', 'EMA5', 'EMA13', 'TRANGE', 'VpR']
# df_non_null_specific = daily_df2.dropna(subset=columns_to_check)
# df_non_null_specific


df_non_null.to_csv(SITE / f'investing/data/preprocessed/{ticker}.csv')


# #### ideas
# - 52 wk Range
# - Most Active Stocks, Top Gainers, Top Losers.
# - the difference between SMAs in different periods or the difference between SMA and EMA in the same period
#
# - component's : 
#     - change | avg change | 
#     - avg price (daily or in a period) | 
#     - avg vol (daily or in a period) | 
#     - performance (common range or in a period) | compare with index (by multi timeframe)
#     - fundametal: marketcap | revenue | P/E | Beta

