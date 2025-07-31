import seaborn as sns
from utils import ForexPreprocessor
sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')
get_ipython().run_line_magic('matplotlib', 'inline')
from data_exploration import explore
timeframe = 60
processor = ForexPreprocessor(timeframe)
data = processor.load_and_preprocess(f'data/GBPUSD/GBPUSD_{timeframe}.csv')
data.info()
data.head()
str_var_list, num_var_list, all_var_list = explore.get_dtypes(data=data)
print(str_var_list)
print(num_var_list)
print(all_var_list)
explore.describe(data=data, output_path='./output/')
explore.discrete_var_barplot(x='trading_session', y='volume_ratio', data=
    data, output_path='./output/')
explore.discrete_var_countplot(x='price_trend', data=data, output_path=
    './output/')
explore.discrete_var_boxplot(x='hour', y='change', data=data, output_path=
    './output/')
explore.discrete_var_boxplot(x='trading_session', y='volatility', data=data,
    output_path='./output/')
explore.discrete_var_boxplot(x='trading_session', y='change', data=data,
    output_path='./output/')
explore.discrete_var_boxplot(x='rsi_zone', y='future_change', data=data,
    output_path='./output/')
explore.discrete_var_boxplot(x='volatility_level', y='future_change', data=
    data, output_path='./output/')
explore.discrete_var_boxplot(x='volume_level', y='future_change', data=data,
    output_path='./output/')
data['price_trend_code'] = data['price_trend'].cat.codes
data['session_code'] = data['trading_session'].cat.codes
data['vol_level_code'] = data['volatility_level'].cat.codes
explore.discrete_var_boxplot(x='trend_position', y='future_change', data=
    data, output_path='./output/')
explore.continuous_var_distplot(x=data['rsi'], output_path='./output/', bins=50
    )
explore.continuous_var_distplot(x=data['volatility'], output_path=
    './output/', bins=30)
explore.scatter_plot(x=processor.lag_columns[0], y='future_change', data=
    data, output_path='./output/')
explore.scatter_plot(x='rsi', y='future_change', data=data, output_path=
    './output/')
explore.correlation_plot(data=data[processor.momentum_features],
    output_path='./output/')
session_trend_pivot = data.pivot_table(values='change', index=
    'trading_session', columns='price_trend', aggfunc='mean')
explore.heatmap(data=session_trend_pivot, output_path='./output/', fmt='.3f')
rsi_vol_pivot = data.pivot_table(values='future_change', index='rsi_zone',
    columns='volatility_level', aggfunc='mean')
explore.heatmap(data=rsi_vol_pivot, output_path='./output/', fmt='.3f')
hourly_target_pivot = data.pivot_table(values='target', index='hour',
    columns='day_of_week', aggfunc='mean')
explore.heatmap(data=hourly_target_pivot, output_path='./output/', fmt='.3f')
pivot_data = data.pivot_table(values='change', index='hour', columns=
    'day_of_week', aggfunc='mean')
explore.heatmap(data=pivot_data, output_path='./output/', fmt='.2f')