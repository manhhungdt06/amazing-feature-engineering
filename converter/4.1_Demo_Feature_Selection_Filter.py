# Generated from: 4.1_Demo_Feature_Selection_Filter.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from feature_selection import filter_method as ft


# ## Load Dataset


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']],
                  columns= np.append(data['feature_names'], ['target']))


data.head(5)


X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['target'], axis=1), 
                                                    data.target, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape


# ## Variance method
# removing features that show the same value for the majority/all of the observations (constant/quasi-constant features)


# the original dataset has no constant variable
quasi_constant_feature = ft.constant_feature_detect(data=X_train,threshold=0.9)


# lets create a duumy variable that help us do the demonstration
X_train['dummy'] = np.floor(X_train['worst smoothness']*10)
# variable dummy has> 92% of the observations show one value, 1.0
X_train.dummy.value_counts() / float(len(X_train))


quasi_constant_feature = ft.constant_feature_detect(data=X_train,threshold=0.9)
quasi_constant_feature


# drop that variable
X_train.drop(labels=quasi_constant_feature,axis=1,inplace=True)
print(X_train.shape)


# ## Correlation method
# remove features that are highly correlated with each other


corr = ft.corr_feature_detect(data=X_train,threshold=0.9)
# print all the correlated feature groups!
for i in corr:
    print(i,'\n')


# then we can decide which ones to remove.


# ## Mutual Information Filter
# Mutual information measures how much information the presence/absence of a feature contributes to making the correct prediction on Y.


# select the top 3 features
mi = ft.mutual_info(X=X_train,y=y_train,select_k=3)
print(mi)


# select the top 20% features
mi = ft.mutual_info(X=X_train,y=y_train,select_k=0.2)
print(mi)


# ## Chi-Square Filter
# Compute chi-squared stats between each non-negative feature and class


# select the top 3 features
chi = ft.chi_square_test(X=X_train,y=y_train,select_k=3)
print(chi)


# select the top 20% features
chi = ft.chi_square_test(X=X_train,y=y_train,select_k=0.2)
print(chi)


# ## Univariate ROC-AUC or MSE
# builds one decision tree per feature, to predict the target, then make predictions and ranks the features according to the machine learning metric (roc-auc or mse)


uni_roc_auc = ft.univariate_roc_auc(X_train=X_train,y_train=y_train,
                                   X_test=X_test,y_test=y_test,threshold=0.8)
print(uni_roc_auc)


uni_mse = ft.univariate_mse(X_train=X_train,y_train=y_train,
                            X_test=X_test,y_test=y_test,threshold=0.4)
print(uni_mse)

