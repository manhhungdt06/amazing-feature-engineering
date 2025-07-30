import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_palette("colorblind")
plt.style.use("default")


def get_dtypes(data, drop_col=[]):
    name_of_col = list(data.columns)
    num_var_list = []
    str_var_list = name_of_col.copy()
    for var in name_of_col:
        if data[var].dtypes in (int, np.int64, np.uint, np.int32, float, np.float64, np.float32, np.double):
            str_var_list.remove(var)
            num_var_list.append(var)
    for var in drop_col:
        if var in str_var_list:
            str_var_list.remove(var)
        if var in num_var_list:
            num_var_list.remove(var)
    all_var_list = str_var_list + num_var_list
    return str_var_list, num_var_list, all_var_list


def describe(data, output_path=None):
    result = data.describe(include="all")
    if output_path is not None:
        output = os.path.join(output_path, "describe.csv")
        result.to_csv(output)
    return result


def discrete_var_barplot(x, y, data, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, "Barplot_" +
                              str(x) + "_" + str(y) + ".png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def discrete_var_countplot(x, data, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=x, data=data)
    if output_path is not None:
        output = os.path.join(output_path, "Countplot_" + str(x) + ".png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def discrete_var_boxplot(x, y, data, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, "Boxplot_" +
                              str(x) + "_" + str(y) + ".png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def continuous_var_distplot(x, output_path=None, bins=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=x, kde=False, bins=bins)
    if output_path is not None:
        output = os.path.join(output_path, "Distplot_" + str(x.name) + ".png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def scatter_plot(x, y, data, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=data)
    if output_path is not None:
        x_name = x if isinstance(x, str) else x.name
        y_name = y if isinstance(y, str) else y.name
        output = os.path.join(output_path, "Scatter_plot_" +
                              str(x_name) + "_" + str(y_name) + ".png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def correlation_plot(data, output_path=None):
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corrmat, cmap="YlGnBu", linewidths=0.5, annot=True)
    if output_path is not None:
        output = os.path.join(output_path, "Corr_plot.png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()


def heatmap(data, output_path=None, fmt="d"):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(data, cmap="YlGnBu", linewidths=0.5, annot=True, fmt=fmt)
    if output_path is not None:
        output = os.path.join(output_path, "Heatmap.png")
        plt.savefig(output, dpi=100, bbox_inches='tight')
    plt.close()
