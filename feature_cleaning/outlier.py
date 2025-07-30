import pandas as pd
import numpy as np


def outlier_detect_arbitrary(data, col, upper_fence, lower_fence):
    para = (upper_fence, lower_fence)
    tmp = pd.concat([data[col] > upper_fence, data[col] < lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    return outlier_index, para


def outlier_detect_IQR(data, col, threshold=3):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    return outlier_index, para


def outlier_detect_mean_std(data, col, threshold=3):
    Upper_fence = data[col].mean() + threshold * data[col].std()
    Lower_fence = data[col].mean() - threshold * data[col].std()
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    return outlier_index, para


def outlier_detect_MAD(data, col, threshold=3.5):
    median = data[col].median()
    median_absolute_deviation = np.median(
        [np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series(
        [0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    return outlier_index


def impute_outlier_with_arbitrary(data, outlier_index, value, col=[]):
    data_copy = data.copy()
    for i in col:
        data_copy.loc[outlier_index, i] = value
    return data_copy


def windsorization(data, col, para, strategy='both'):
    data_copy = data.copy()
    if strategy == 'both':
        data_copy.loc[data_copy[col] > para[0], col] = para[0]
        data_copy.loc[data_copy[col] < para[1], col] = para[1]
    elif strategy == 'top':
        data_copy.loc[data_copy[col] > para[0], col] = para[0]
    elif strategy == 'bottom':
        data_copy.loc[data_copy[col] < para[1], col] = para[1]
    return data_copy


def drop_outlier(data, outlier_index):
    return data[~outlier_index]


def impute_outlier_with_avg(data, col, outlier_index, strategy='mean'):
    data_copy = data.copy()
    if strategy == 'mean':
        data_copy.loc[outlier_index, col] = data_copy[col].mean()
    elif strategy == 'median':
        data_copy.loc[outlier_index, col] = data_copy[col].median()
    elif strategy == 'mode':
        data_copy.loc[outlier_index, col] = data_copy[col].mode()[0]
    return data_copy
