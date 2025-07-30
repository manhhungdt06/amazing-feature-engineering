# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab


def diagnostic_plots(df, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)
    plt.show()


def log_transform(data, cols=[]):
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i + '_log'] = np.log(data_copy[i] + 1)
        diagnostic_plots(data_copy, i + '_log')
    return data_copy


def reciprocal_transform(data, cols=[]):
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i + '_reciprocal'] = 1 / data_copy[i]
        diagnostic_plots(data_copy, i + '_reciprocal')
    return data_copy


def square_root_transform(data, cols=[]):
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i + '_square_root'] = np.sqrt(data_copy[i])
        diagnostic_plots(data_copy, i + '_square_root')
    return data_copy


def exp_transform(data, coef, cols=[]):
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i + '_exp'] = data_copy[i] ** coef
        diagnostic_plots(data_copy, i + '_exp')
    return data_copy
