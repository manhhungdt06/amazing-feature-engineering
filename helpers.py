import sys
import numpy as np


class LazyCallable(object):
    def __init__(self, name):
        self.n, self.f = name, None
    
    def __call__(self, *a, **k):
        if self.f is None:
            modn, funcn = self.n.rsplit('.', 1)
            if modn not in sys.modules:
                __import__(modn)
                self.f = getattr(sys.modules[modn], funcn)
            else:
                self.f = getattr(sys.modules[modn], funcn)
        return self.f(*a, **k)


def add_percentage_change(df, column_name, periods):
    period_map = {'W': 5,'M': 21, '3M': 63, 'Y': 252, '3Y': 252 * 3}
    df['StartOfYear'] = df.groupby(df.index.year)[column_name].transform('first')
    for period in periods:
        if period == 'YTD':
            df['YTD'] = (df[column_name] / df['StartOfYear'] - 1) * 100
        else:
            period_value = period_map.get(period, period)
            new_column_name = f'Chg{period}'
            df[new_column_name] = df[column_name].pct_change(periods=period_value) * 100
    df.drop(columns=['StartOfYear'], inplace=True)
    return df


def add_rolling_functions(df, column_names, window_sizes, functions):
    for column_name in column_names:
        for window_size in window_sizes:
            if isinstance(window_size, str):
                window = window_size
            elif isinstance(window_size, int):
                window = window_size
            else:
                raise ValueError("Window size must be either a string (e.g., '8D') or an integer (e.g., 8)")
            
            for func in functions:
                if func == 'mean':
                    df[f'{column_name}Mean{window}'] = df[column_name].rolling(window=window).mean()
                elif func == 'sum':
                    df[f'{column_name}Sum{window}'] = df[column_name].rolling(window=window).sum()
                elif func == 'max':
                    df[f'{column_name}Max{window}'] = df[column_name].rolling(window=window).max()
                elif func == 'min':
                    df[f'{column_name}Min{window}'] = df[column_name].rolling(window=window).min()                
                elif func == 'var':
                    df[f'{column_name}Var{window}'] = df[column_name].rolling(window=window).var()
                elif func == 'std':
                    df[f'{column_name}Std{window}'] = df[column_name].rolling(window=window).std()
                elif func == 'skew':
                    df[f'{column_name}Skew{window}'] = df[column_name].rolling(window=window).skew()
                elif func == 'kurt':
                    df[f'{column_name}Kurt{window}'] = df[column_name].rolling(window=window).kurt()
                elif func == 'shift':
                    df[f'{column_name}Shift{window}'] = df[column_name].rolling(window=window).shift()
                elif func == 'diff':
                    df[f'{column_name}Diff{window}'] = df[column_name].rolling(window=window).diff()
                else:
                    raise ValueError(f"Unsupported function: {func}")
                    
    return df


def add_technical_indicators(df, indicators):
    for indicator, params in indicators.items():
        time_periods = params.get('time_periods', [])
        input_columns = params.get('input_columns', [])
        if isinstance(input_columns, str):
            input_columns = [input_columns]  # Convert single input column to list

        if not isinstance(time_periods, list) or time_periods == "":
            time_periods = [""]

        for time_period in time_periods:
            column_name = f'{indicator}{time_period}'
            indicator_func = LazyCallable(f'talib.{indicator}')
            if time_period:
                df[column_name] = indicator_func(*[df[col] for col in input_columns], timeperiod=time_period)
            else:
                df[column_name] = indicator_func(*[df[col] for col in input_columns])
    
    return df


# https://github.com/jasonstrimpel/volatility-trading
def calculate_close_to_close_volatility(df, close_, windows=[30,], trading_periods=[252,], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            df[f'log_return_{trading_period}_{window}'] = np.log(df[close_] / df[close_].shift(1))
            df[f'c_vol_{trading_period}_{window}'] = df[f'log_return_{trading_period}_{window}'].rolling(window=window).std() * np.sqrt(trading_period) * 100
            df.drop(columns=[f'log_return_{trading_period}_{window}'], inplace=True)
    
    if clean:
        df = df.dropna()
    
    return df


def calculate_parkinson_volatility(df, high_, low_, windows=[30,], trading_periods=[252,], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            rs = (1.0 / (4.0 * np.log(2.0))) * ((df[high_] / df[low_]).apply(np.log)) ** 2.0
            
            def f(v):
                return (trading_period * v.mean()) ** 0.5
            
            result_name = f'p_vol_{trading_period}_{window}'
            df[result_name] = rs.rolling(window=window, center=False).apply(func=f) * 100
    
    if clean:
        df = df.dropna()
    
    return df


def calculate_garman_klass_volatility(df, high_, low_, close_, open_,  windows=[30,], trading_periods=[252,], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            log_hl = np.log(df[high_] / df[low_])
            log_co = np.log(df[close_] / df[open_])
            rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
            
            def f(v):
                return (trading_period * v.mean()) ** 0.5
            
            result_col_name = f'gk_vol_{trading_period}_{window}'
            df[result_col_name] = rs.rolling(window=window, center=False).apply(func=f) * 100
    
    if clean:
        df = df.dropna()
    
    return df


def calculate_hodges_tompkins_volatility(df, close_, windows=[30], trading_periods=[252], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            log_return = np.log(df[close_] / df[close_].shift(1))
            vol = log_return.rolling(window=window, center=False).std() * np.sqrt(trading_period)
            h = window
            n = (log_return.count() - h) + 1
            adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))
            df[f'ht_vol_{trading_period}_{window}'] = vol * adj_factor * 100
    
    if clean:
        df = df.dropna()
    
    return df


def calculate_rogers_satchell_volatility(df, high_, low_, close_, open_, windows=[30], trading_periods=[252], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            log_ho = np.log(df[high_] / df[open_])
            log_lo = np.log(df[low_] / df[open_])
            log_co = np.log(df[close_] / df[open_])
            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

            def f(v):
                return (trading_period * v.mean()) ** 0.5
            
            df[f'rs_vol_{trading_period}_{window}'] = rs.rolling(window=window, center=False).apply(func=f) * 100
    
    if clean:
        df = df.dropna()
    
    return df


def calculate_yang_zhang_volatility(df, high_, low_ , close_, open_, windows=[30], trading_periods=[252], clean=True):
    for trading_period in trading_periods:
        for window in windows:
            log_ho = np.log(df[high_] / df[open_])
            log_lo = np.log(df[low_] / df[open_])
            log_co = np.log(df[close_] / df[open_])
            
            log_oc = np.log(df[open_] / df[close_].shift(1))
            log_oc_sq = log_oc ** 2
            
            log_cc = np.log(df[close_] / df[close_].shift(1))
            log_cc_sq = log_cc ** 2
            
            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            
            close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
            open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
            window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
            
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            
            df[f'yz_vol_{trading_period}_{window}'] = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_period) * 100
    
    if clean:
        df = df.dropna()
    
    return df
