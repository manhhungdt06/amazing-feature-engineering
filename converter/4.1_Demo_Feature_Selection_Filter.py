import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from feature_selection import filter_method as ft
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(
    data['feature_names'], ['target']))
data.head(5)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'target'], axis=1), data.target, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
quasi_constant_feature = ft.constant_feature_detect(data=X_train, threshold=0.9
    )
X_train['dummy'] = np.floor(X_train['worst smoothness'] * 10)
X_train.dummy.value_counts() / float(len(X_train))
quasi_constant_feature = ft.constant_feature_detect(data=X_train, threshold=0.9
    )
quasi_constant_feature
X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
print(X_train.shape)
corr = ft.corr_feature_detect(data=X_train, threshold=0.9)
for i in corr:
    print(i, '\n')
mi = ft.mutual_info(X=X_train, y=y_train, select_k=3)
print(mi)
mi = ft.mutual_info(X=X_train, y=y_train, select_k=0.2)
print(mi)
chi = ft.chi_square_test(X=X_train, y=y_train, select_k=3)
print(chi)
chi = ft.chi_square_test(X=X_train, y=y_train, select_k=0.2)
print(chi)
uni_roc_auc = ft.univariate_roc_auc(X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test, threshold=0.8)
print(uni_roc_auc)
uni_mse = ft.univariate_mse(X_train=X_train, y_train=y_train, X_test=X_test,
    y_test=y_test, threshold=0.4)
print(uni_mse)