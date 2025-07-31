import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(
    data['feature_names'], ['target']))
data.head(5)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'target'], axis=1), data.target, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
sfs1 = SFS(RandomForestClassifier(n_jobs=-1, n_estimators=5), k_features=10,
    forward=True, floating=False, verbose=1, scoring='roc_auc', cv=3)
sfs1 = sfs1.fit(np.array(X_train), y_train)
selected_feat1 = X_train.columns[list(sfs1.k_feature_idx_)]
selected_feat1
sfs2 = SFS(RandomForestClassifier(n_jobs=-1, n_estimators=5), k_features=10,
    forward=False, floating=False, verbose=1, scoring='roc_auc', cv=3)
sfs2 = sfs1.fit(np.array(X_train.fillna(0)), y_train)
selected_feat2 = X_train.columns[list(sfs2.k_feature_idx_)]
selected_feat2
efs1 = EFS(RandomForestClassifier(n_jobs=-1, n_estimators=5, random_state=0
    ), min_features=1, max_features=6, scoring='roc_auc', print_progress=
    True, cv=2)
efs1 = efs1.fit(np.array(X_train[X_train.columns[0:10]].fillna(0)), y_train)
selected_feat3 = X_train.columns[list(efs1.best_idx_)]
selected_feat3