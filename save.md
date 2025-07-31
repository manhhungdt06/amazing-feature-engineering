```python

from feature_cleaning import outlier as ot

timeframe = 60
processor = ForexPreprocessor(timeframe)
data = processor.load_and_preprocess(f'data/GBPUSD/GBPUSD_{timeframe}.csv')

outlier_cols = ['change', 'volume_ratio', 'rsi', 'volatility']

for col in outlier_cols:
   print(f"\n--- {col} Outlier Analysis ---")
   print(f"Min: {data[col].min():.4f}, Max: {data[col].max():.4f}")
   
   iqr_index, iqr_para = ot.outlier_detect_IQR(data=data, col=col, threshold=2.5)
   print(f"IQR outliers: {iqr_index.sum()}, Upper: {iqr_para[0]:.4f}, Lower: {iqr_para[1]:.4f}")
   
   mad_index = ot.outlier_detect_MAD(data=data, col=col, threshold=3.5)
   print(f"MAD outliers: {mad_index.sum()}")

data_winsorized = data.copy()
for col in ['change', 'volume_ratio']:
   index, para = ot.outlier_detect_IQR(data=data, col=col, threshold=2.5)
   data_winsorized = ot.windsorization(data=data_winsorized, col=col, para=para, strategy='both')

data_cleaned = data.copy()
volume_index, _ = ot.outlier_detect_IQR(data=data, col='volume_ratio', threshold=3)
data_cleaned = ot.drop_outlier(data=data_cleaned, outlier_index=volume_index)

rsi_index, _ = ot.outlier_detect_arbitrary(data=data, col='rsi', upper_fence=95, lower_fence=5)
data_rsi_capped = ot.windsorization(data=data, col='rsi', para=(95, 5), strategy='both')

```

---

```python
from feature_cleaning import rare_values as ra

timeframe = 60
processor = ForexPreprocessor(timeframe)
data = processor.load_and_preprocess(f'data/GBPUSD/GBPUSD_{timeframe}.csv')

categorical_cols = ['trading_session', 'price_trend', 'volatility_level', 'volume_level', 'rsi_zone']

for col in categorical_cols:
   print(f'\n--- {col} Value Distribution ---')
   proportions = data[col].value_counts(normalize=True)
   print(proportions)
   rare_values = proportions[proportions < 0.05]
   if len(rare_values) > 0:
       print(f"Rare values (< 5%): {rare_values.index.tolist()}")

enc_grouping = ra.GroupingRareValues(cols=['rsi_zone', 'volatility_level'], threshold=0.03).fit(data)
data_grouped = enc_grouping.transform(data)
print(f"\nAfter grouping rare values:")
print(data_grouped['rsi_zone'].value_counts())

enc_mode = ra.ModeImputation(cols=['volume_level'], threshold=0.05).fit(data)
data_mode = enc_mode.transform(data)
print(f"\nAfter mode imputation:")
print(data_mode['volume_level'].value_counts())

hourly_dist = data['hour'].value_counts(normalize=True).sort_index()
rare_hours = hourly_dist[hourly_dist < 0.02].index.tolist()
if rare_hours:
   print(f"\nRare trading hours: {rare_hours}")
```