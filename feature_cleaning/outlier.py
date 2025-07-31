import pandas as pd
import numpy as np
from scipy import stats

def outlier_detect_arbitrary(data, col, upper_fence, lower_fence):
    para = (upper_fence, lower_fence)
    outlier_index = (data[col] > upper_fence) | (data[col] < lower_fence)
    return outlier_index, para

def outlier_detect_IQR(data, col, threshold=1.5):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - (IQR * threshold)
    upper_fence = Q3 + (IQR * threshold)
    para = (upper_fence, lower_fence)
    outlier_index = (data[col] > upper_fence) | (data[col] < lower_fence)
    return outlier_index, para

def outlier_detect_zscore(data, col, threshold=3):
    z_scores = np.abs(stats.zscore(data[col].dropna()))
    outlier_index = pd.Series(False, index=data.index)
    outlier_index.iloc[data[col].dropna().index] = z_scores > threshold
    return outlier_index

def outlier_detect_modified_zscore(data, col, threshold=3.5):
    median = data[col].median()
    mad = np.median(np.abs(data[col] - median))
    modified_z_scores = 0.6745 * (data[col] - median) / mad
    outlier_index = np.abs(modified_z_scores) > threshold
    return outlier_index

def outlier_detect_isolation_forest(data, col, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_pred = iso_forest.fit_predict(data[[col]].dropna())
    outlier_index = pd.Series(False, index=data.index)
    outlier_index.iloc[data[col].dropna().index] = outlier_pred == -1
    return outlier_index

def winsorize_percentile(data, col, lower_percentile=5, upper_percentile=95):
    data_copy = data.copy()
    lower_val = data[col].quantile(lower_percentile/100)
    upper_val = data[col].quantile(upper_percentile/100)
    data_copy[col] = np.clip(data_copy[col], lower_val, upper_val)
    return data_copy

def outlier_summary(data, cols=None):
    if cols is None:
        cols = data.select_dtypes(include=[np.number]).columns
    
    summary = []
    for col in cols:
        iqr_outliers, _ = outlier_detect_IQR(data, col)
        zscore_outliers = outlier_detect_zscore(data, col)
        mad_outliers = outlier_detect_modified_zscore(data, col)
        
        summary.append({
            'column': col,
            'total_values': len(data[col].dropna()),
            'iqr_outliers': iqr_outliers.sum(),
            'zscore_outliers': zscore_outliers.sum(),
            'mad_outliers': mad_outliers.sum(),
            'min': data[col].min(),
            'max': data[col].max()
        })
    
    return pd.DataFrame(summary)