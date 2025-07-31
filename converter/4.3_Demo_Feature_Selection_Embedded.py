import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from feature_selection import embedded_method
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(
    data['feature_names'], ['target']))
data.head(5)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'target'], axis=1), data.target, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
scaler = RobustScaler()
scaler.fit(X_train)
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2'))
sel_.fit(scaler.transform(X_train), y_train)
selected_feat = X_train.columns[sel_.get_support()]
print('total features: {}'.format(X_train.shape[1]))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.
    estimator_.coef_ == 0)))
removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
removed_feats
X_train_selected = sel_.transform(X_train.fillna(0))
X_test_selected = sel_.transform(X_test.fillna(0))
X_train_selected.shape, X_test_selected.shape
model = embedded_method.rf_importance(X_train=X_train, y_train=y_train,
    max_depth=10, top_n=10)
from sklearn.feature_selection import SelectFromModel
feature_selection = SelectFromModel(model, threshold=0.05, prefit=True)
selected_feat = X_train.columns[feature_selection.get_support()]
selected_feat
feature_selection2 = SelectFromModel(model, threshold='2*median', prefit=True)
selected_feat2 = X_train.columns[feature_selection2.get_support()]
selected_feat2
model = embedded_method.gbt_importance(X_train=X_train, y_train=y_train,
    max_depth=10, top_n=10)
feature_selection = SelectFromModel(model, threshold=0.01, prefit=True)
selected_feat = X_train.columns[feature_selection.get_support()]
selected_feat