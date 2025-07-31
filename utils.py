import pandas as pd
import numpy as np


class ForexPreprocessor:
   def __init__(self, timeframe):
       self.timeframe = timeframe
       self.lag_periods = {60: [1, 6, 24], 240: [1, 3, 7], 1440: [1, 2, 5]}
       self.lag_suffixes = {60: ['1h', '6h', '24h'], 240: ['4h', '12h', '28h'], 1440: ['1d', '2d', '5d']}
       self.session_bins = {60: [0, 6, 14, 22, 24], 240: [0, 8, 16, 24], 1440: [0, 12, 24]}
       self.session_labels = {60: ['Asian', 'European', 'American', 'Pacific'], 240: ['Asian', 'European', 'American'], 1440: ['AM', 'PM']}
       self.window_size = max(20, timeframe // 60)
       
   def load_and_preprocess(self, file_path, drop_na=True):
        use_cols = ['time', 'close', 'volume']
        data = pd.read_csv(file_path, usecols=use_cols, parse_dates=['time'])
        data['change'] = data['close'].pct_change()*100
        
        data['hour'] = data['time'].dt.hour
        data['day_of_week'] = data['time'].dt.dayofweek
        data['month'] = data['time'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        data['sma_20'] = data['close'].rolling(window=self.window_size).mean()
        data['sma_50'] = data['close'].rolling(window=self.window_size*2).mean()
        data['rsi'] = 100 - (100 / (1 + data['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / 
                                    data['close'].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
        
        data['high_low_pct'] = ((data['close'] - data['close'].rolling(self.window_size).min()) / 
                                (data['close'].rolling(self.window_size).max() - data['close'].rolling(self.window_size).min())) * 100
        data['volatility'] = data['change'].rolling(self.window_size).std()
        data['price_position'] = (data['close'] > data['sma_20']).astype(int)
        
        data['volume_ma'] = data['volume'].rolling(self.window_size).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        for i, lag in enumerate(self.lag_periods[self.timeframe]):
            data[f'change_{self.lag_suffixes[self.timeframe][i]}'] = data['change'].shift(lag)
        
        data['future_change'] = data['change'].shift(-1)
        
        data['price_trend'] = pd.cut(data['change'], bins=[-np.inf, -0.5, 0.5, np.inf], 
                                    labels=['Down', 'Flat', 'Up'])
        data['volatility_level'] = pd.cut(data['volatility'], bins=3, labels=['Low', 'Medium', 'High'])
        data['volume_level'] = pd.cut(data['volume_ratio'], bins=[0, 0.8, 1.2, np.inf], 
                                    labels=['Low', 'Normal', 'High'])
        data['rsi_zone'] = pd.cut(data['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
        
        data['trading_session'] = pd.cut(data['hour'], bins=self.session_bins[self.timeframe], 
                                        labels=self.session_labels[self.timeframe], include_lowest=True)
        
        data['trend_position'] = ((data['price_position'] == 1) & (data['price_trend'] == 'Up')).astype('category')
        data['target'] = (data['future_change'] > 0).astype(int)
        
        categorical_features = ['price_trend', 'volatility_level', 'volume_level', 'trading_session', 'rsi_zone']
        for col in categorical_features:
            data[col] = data[col].astype('category')
        if drop_na:
            data.dropna(inplace=True)
        return data
   
   @property
   def lag_columns(self):
       return [f'change_{suffix}' for suffix in self.lag_suffixes[self.timeframe]]
   
   @property
   def momentum_features(self):
       return ['change'] + self.lag_columns + ['future_change', 'rsi', 'volatility', 'volume_ratio', 'price_position', 'target']