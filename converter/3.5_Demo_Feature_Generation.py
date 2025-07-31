import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
    test_size=0.3, random_state=0)
X_train.shape, X_test.shape
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False).fit(X_train[['Pclass',
    'SibSp']])
tmp = pf.transform(X_train[['Pclass', 'SibSp']])
X_train_copy = pd.DataFrame(tmp, columns=pf.get_feature_names_out(['Pclass',
    'SibSp']))
print(X_train_copy.head(6))
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
gbdt = GradientBoostingClassifier(n_estimators=20)
one_hot = OneHotEncoder()
X_train = X_train[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
gbdt.fit(X_train, y_train)
X_leaf_index = gbdt.apply(X_train)[:, :, 0]
print(, X_leaf_index)
one_hot.fit(X_leaf_index)
X_one_hot = one_hot.transform(X_leaf_index)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_one_hot, y_train)
y_pred = lr.predict_proba(one_hot.transform(gbdt.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for GBDT derived feature + LR：', roc_auc_score(y_test, y_pred))
rf = RandomForestClassifier(n_estimators=20)
one_hot = OneHotEncoder(handle_unknown='ignore')
X_train = X_train[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
rf.fit(X_train, y_train)
X_leaf_index = rf.apply(X_train)
print(, X_leaf_index)
one_hot.fit(X_leaf_index)
X_one_hot = one_hot.transform(X_leaf_index)
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_one_hot, y_train)
y_pred = lr.predict_proba(one_hot.transform(rf.apply(X_test)))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for RandomForest derived feature + LR：', roc_auc_score(y_test,
    y_pred))
from scipy.sparse import hstack
X_train_ext = hstack([one_hot.transform(gbdt.apply(X_train)[:, :, 0]), X_train]
    )
X_test_ext = hstack([one_hot.transform(gbdt.apply(X_test)[:, :, 0]), X_test])
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train_ext, y_train)
y_pred = lr.predict_proba(X_test_ext)[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for GBDT derived feature + Raw feature +LR：', roc_auc_score(
    y_test, y_pred))
X_train_ext = hstack([one_hot.transform(rf.apply(X_train)), X_train])
X_test_ext = hstack([one_hot.transform(rf.apply(X_test)), X_test])
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train_ext, y_train)
y_pred = lr.predict_proba(X_test_ext)[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for RandomForest derived feature + Raw feature + LR：',
    roc_auc_score(y_test, y_pred))
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict_proba(X_test)[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for RandomForest derived feature + LR：', roc_auc_score(y_test,
    y_pred))
gbdt = GradientBoostingClassifier(n_estimators=20)
X_train = X_train[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict_proba(X_test)[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for Raw feature + GBDT：', roc_auc_score(y_test, y_pred))
rf = RandomForestClassifier(n_estimators=20)
X_train = X_train[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[['Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print('AUC for Raw feature + RF：', roc_auc_score(y_test, y_pred))