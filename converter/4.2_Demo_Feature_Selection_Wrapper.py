# Generated from: 4.2_Demo_Feature_Selection_Wrapper.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# plt.style.use('seaborn-colorblind')
# %matplotlib inline
# from feature_selection import filter_method as ft


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


# ## Forward Selection
#


# step forward feature selection
# select top 10 features based on the optimal roc_auc and RandomForest Classifier

sfs1 = SFS(RandomForestClassifier(n_jobs=-1,n_estimators=5), 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=1,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)


selected_feat1= X_train.columns[list(sfs1.k_feature_idx_)]
selected_feat1


# ## Backward Elimination


# step backward feature selection
# select top 10 features based on the optimal roc_auc and RandomForest Classifier

sfs2 = SFS(RandomForestClassifier(n_jobs=-1,n_estimators=5), 
           k_features=10, 
           forward=False, 
           floating=False, 
           verbose=1,
           scoring='roc_auc',
           cv=3)

sfs2 = sfs1.fit(np.array(X_train.fillna(0)), y_train)


selected_feat2= X_train.columns[list(sfs2.k_feature_idx_)]
selected_feat2



# Note that SFS and SBE return different results


# ## Exhaustive Feature Selection


efs1 = EFS(RandomForestClassifier(n_jobs=-1,n_estimators=5, random_state=0), 
           min_features=1,
           max_features=6, 
           scoring='roc_auc',
           print_progress=True,
           cv=2)

# in order to shorter search time for the demonstration
# we only try all possible 1,2,3,4,5,6
# feature combinations from a dataset of 10 features

efs1 = efs1.fit(np.array(X_train[X_train.columns[0:10]].fillna(0)), y_train)


selected_feat3= X_train.columns[list(efs1.best_idx_)]
selected_feat3

