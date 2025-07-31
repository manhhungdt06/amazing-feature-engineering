import pandas as pd
import numpy as np
from warnings import warn


def check_missing(data, output_path=None):
    result = pd.concat([data.isnull().sum(), data.isnull().mean()], axis=1)
    result = result.rename(columns={0: "total missing", 1: "proportion"})
    if output_path is not None:
        result.to_csv(output_path + "missing.csv")
    return result


def drop_missing(data, axis=0):
    return data.dropna(axis=axis)


def add_var_denote_NA(data, NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + "_is_NA"] = np.where(data_copy[i].isnull(), 1, 0)
        else:
            warn("Column %s has no missing cases" % i)
    return data_copy


def impute_NA_with_arbitrary(data, impute_value, NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i] = data_copy[i].fillna(impute_value)
        else:
            warn("Column %s has no missing cases" % i)
    return data_copy


def impute_NA_with_avg(data, strategy="mean", NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if strategy == "mean":
                data_copy[i] = data_copy[i].fillna(data[i].mean())
            elif strategy == "median":
                data_copy[i] = data_copy[i].fillna(data[i].median())
            elif strategy == "mode":
                data_copy[i] = data_copy[i].fillna(data[i].mode()[0])
        else:
            warn("Column %s has no missing" % i)
    return data_copy


def impute_NA_with_end_of_distribution(data, NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i] = data_copy[i].fillna(data[i].mean() + 3 * data[i].std())
        else:
            warn("Column %s has no missing" % i)
    return data_copy


def impute_NA_with_random(data, NA_col=[], random_state=0):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            random_sample = data_copy[i].dropna().sample(
                data_copy[i].isnull().sum(), random_state=random_state, replace=True
            )
            random_sample.index = data_copy[data_copy[i].isnull()].index
            data_copy.loc[data_copy[i].isnull(), i] = random_sample
        else:
            warn("Column %s has no missing" % i)
    return data_copy


def forward_fill_NA(data, NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i] = data_copy[i].fillna(method='ffill')
        else:
            warn("Column %s has no missing" % i)
    return data_copy


def interpolate_NA(data, method='linear', NA_col=[]):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i] = data_copy[i].interpolate(method=method)
        else:
            warn("Column %s has no missing" % i)
    return data_copy


def impute_NA_conditional(data, NA_col=[], condition_col=None, weekend_value=0):
    data_copy = data.copy()
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if condition_col and condition_col in data_copy.columns:
                weekend_mask = pd.to_datetime(data_copy[condition_col]).dt.dayofweek >= 5
                data_copy.loc[weekend_mask & data_copy[i].isnull(), i] = weekend_value
                data_copy[i] = data_copy[i].fillna(data_copy[i].median())
            else:
                data_copy[i] = data_copy[i].fillna(data_copy[i].median())
        else:
            warn("Column %s has no missing" % i)
    return data_copy