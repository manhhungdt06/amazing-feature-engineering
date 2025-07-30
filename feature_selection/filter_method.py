import pandas as pd
# import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, root_mean_squared_error


def constant_feature_detect(data, threshold=0.98):
    data_copy = data.copy(deep=True)
    quasi_constant_feature = [
        feature for feature in data_copy.columns
        if (data_copy[feature].value_counts() / float(len(data_copy))).max() >= threshold
    ]
    print(f"{len(quasi_constant_feature)} variables are found to be almost constant")
    return quasi_constant_feature


def corr_feature_detect(data, threshold=0.8):
    corrmat = data.corr().abs().unstack().sort_values(ascending=False)
    corrmat = corrmat[(corrmat >= threshold) & (corrmat < 1)].reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']

    grouped_feature_ls, correlated_groups = [], []
    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls += list(
                correlated_block.feature2.unique()) + [feature]
            correlated_groups.append(correlated_block)
    return correlated_groups


def mutual_info(X, y, select_k=10):
    selector = SelectKBest(mutual_info_classif, k=select_k) if select_k >= 1 else SelectPercentile(
        mutual_info_classif, percentile=select_k * 100)
    selector.fit(X, y)
    return X.columns[selector.get_support()]


def chi_square_test(X, y, select_k=10):
    selector = SelectKBest(chi2, k=select_k) if select_k >= 1 else SelectPercentile(
        chi2, percentile=select_k * 100)
    selector.fit(X, y)
    return X.columns[selector.get_support()]


def univariate_roc_auc(X_train, y_train, X_test, y_test, threshold):
    roc_values = [
        roc_auc_score(y_test, DecisionTreeClassifier().fit(X_train[feature].to_frame(
        ), y_train).predict_proba(X_test[feature].to_frame())[:, 1])
        for feature in X_train.columns
    ]
    roc_series = pd.Series(
        roc_values, index=X_train.columns).sort_values(ascending=False)
    print(roc_series)
    print(
        f"{len(roc_series[roc_series > threshold])} out of {len(X_train.columns)} features are kept")
    return roc_series[roc_series > threshold]


def univariate_mse(X_train, y_train, X_test, y_test, threshold):
    mse_values = [
        root_mean_squared_error(y_test, DecisionTreeRegressor().fit(
            X_train[feature].to_frame(), y_train).predict(X_test[feature].to_frame()))
        for feature in X_train.columns
    ]
    mse_series = pd.Series(
        mse_values, index=X_train.columns).sort_values(ascending=False)
    print(mse_series)
    print(
        f"{len(mse_series[mse_series > threshold])} out of {len(X_train.columns)} features are kept")
    return mse_series[mse_series > threshold]
