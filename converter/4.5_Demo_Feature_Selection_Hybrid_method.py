import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from feature_selection import hybrid
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(
    data['feature_names'], ['target']))
data.head(5)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'target'], axis=1), data.target, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
sel_ = RFE(RandomForestClassifier(n_estimators=20), n_features_to_select=10)
sel_.fit(X_train.fillna(0), y_train)
selected_feat = X_train.columns[sel_.get_support()]
print(selected_feat)
features_to_keep = hybrid.recursive_feature_elimination_rf(X_train=X_train,
    y_train=y_train, X_test=X_test, y_test=y_test, tol=0.001)
features_to_keep
features_to_keep = hybrid.recursive_feature_addition_rf(X_train=X_train,
    y_train=y_train, X_test=X_test, y_test=y_test, tol=0.001)
features_to_keep