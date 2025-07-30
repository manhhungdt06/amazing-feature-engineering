import pandas as pd
import numpy as np


def load_and_preprocess_forex(file_path, timeframe):
    use_cols = ['time', 'close', 'volume']
    data = pd.read_csv(file_path, usecols=use_cols, parse_dates=['time'])
    data['change'] = data['close'].pct_change()*100
    
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    window_size = max(20, timeframe // 60)
    data['sma_20'] = data['close'].rolling(window=window_size).mean()
    data['sma_50'] = data['close'].rolling(window=window_size*2).mean()
    data['rsi'] = 100 - (100 / (1 + data['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / 
                              data['close'].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
    
    data['high_low_pct'] = ((data['close'] - data['close'].rolling(window_size).min()) / 
                           (data['close'].rolling(window_size).max() - data['close'].rolling(window_size).min())) * 100
    data['volatility'] = data['change'].rolling(window_size).std()
    data['price_position'] = (data['close'] > data['sma_20']).astype(int)
    
    data['volume_ma'] = data['volume'].rolling(window_size).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    lag_periods = {60: [1, 6, 24], 240: [1, 3, 7], 1440: [1, 2, 5]}
    for lag in lag_periods[timeframe]:
        data[f'change_{lag}d'] = data['change'].shift(lag)
    
    data['future_change'] = data['change'].shift(-1)
    
    data['price_trend'] = pd.cut(data['change'], bins=[-np.inf, -0.5, 0.5, np.inf], 
                               labels=['Down', 'Flat', 'Up'])
    data['volatility_level'] = pd.cut(data['volatility'], bins=3, labels=['Low', 'Medium', 'High'])
    data['volume_level'] = pd.cut(data['volume_ratio'], bins=[0, 0.8, 1.2, np.inf], 
                                labels=['Low', 'Normal', 'High'])
    data['rsi_zone'] = pd.cut(data['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
    
    session_bins = {60: [0, 6, 14, 22, 24], 240: [0, 8, 16, 24], 1440: [0, 12, 24]}
    session_labels = {60: ['Asian', 'European', 'American', 'Pacific'], 240: ['Asian', 'European', 'American'], 1440: ['AM', 'PM']}
    
    data['trading_session'] = pd.cut(data['hour'], bins=session_bins[timeframe], 
                                   labels=session_labels[timeframe], include_lowest=True)
    
    data['trend_position'] = ((data['price_position'] == 1) & (data['price_trend'] == 'Up')).astype('category')
    data['target'] = (data['future_change'] > 0).astype(int)
    
    categorical_features = ['price_trend', 'volatility_level', 'volume_level', 'trading_session', 'rsi_zone']
    for col in categorical_features:
        data[col] = data[col].astype('category')
    
    data.dropna(inplace=True)
    return data