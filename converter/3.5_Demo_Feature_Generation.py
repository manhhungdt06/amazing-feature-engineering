# Generated from: 3.5_Demo_Feature_Generation.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,  roc_auc_score

# plt.style.use('seaborn-colorblind')
# %matplotlib inline
#from feature_cleaning import rare_values as ra


# ## Load Dataset


use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]

data = pd.read_csv('./data/titanic.csv', usecols=use_cols)



data.head(3)


# Note that we include target variable in the X_train 
# because we need it to supervise our discretization
# this is not the standard way of using train-test-split
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# ## Polynomial Expansion
#
# generate a new feature set consisting of all polynomial combinations of the features with degree less than or equal to the specified degree


# create polynomial combinations of feature 'Pclass','SibSp' with degree 2
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2,include_bias=False).fit(X_train[['Pclass','SibSp']])
tmp = pf.transform(X_train[['Pclass','SibSp']])
# X_train_copy = pd.DataFrame(tmp,columns=pf.get_feature_names(['Pclass','SibSp']))
X_train_copy = pd.DataFrame(tmp,columns=pf.get_feature_names_out(['Pclass','SibSp']))
print(X_train_copy.head(6))


# ## Feature Learning by Trees
# GBDT derived feature + LR


from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

gbdt = GradientBoostingClassifier(n_estimators=20)
one_hot = OneHotEncoder()
# one_hot = OneHotEncoder(handle_unknown='ignore')

X_train = X_train[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)

gbdt.fit(X_train, y_train)

X_leaf_index = gbdt.apply(X_train)[:, :, 0]  # apply return the node index on each tree 
print("sample's belonging node of each base tree \n'",X_leaf_index)
# fit one-hot encoder
one_hot.fit(X_leaf_index)   
X_one_hot = one_hot.transform(X_leaf_index)  


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_one_hot,y_train)
y_pred = lr.predict_proba(one_hot.transform(gbdt.apply(X_test)[:, :, 0]))[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for GBDT derived feature + LR：", roc_auc_score(y_test, y_pred))



# ## Feature Learning by Trees
# RandomForest derived feature + LR


rf = RandomForestClassifier(n_estimators=20)
one_hot = OneHotEncoder(handle_unknown='ignore')

X_train = X_train[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)

rf.fit(X_train, y_train)

X_leaf_index = rf.apply(X_train)  # apply return the node index on each tree 
print("sample's belonging node of each base tree \n'",X_leaf_index)
# fit one-hot encoder
one_hot.fit(X_leaf_index)   
X_one_hot = one_hot.transform(X_leaf_index)  


lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_one_hot,y_train)
y_pred = lr.predict_proba(
    one_hot.transform(rf.apply(X_test)))[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for RandomForest derived feature + LR：", roc_auc_score(y_test, y_pred))



# ##  Feature Learning by Trees
# GBDT derived feature + Raw feature +LR


from scipy.sparse import hstack

X_train_ext = hstack([one_hot.transform(gbdt.apply(X_train)[:, :, 0]), X_train])
X_test_ext = hstack([one_hot.transform(gbdt.apply(X_test)[:, :, 0]), X_test])
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train_ext,y_train)
y_pred = lr.predict_proba(X_test_ext)[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for GBDT derived feature + Raw feature +LR：", roc_auc_score(y_test, y_pred))



# ##  Feature Learning by Trees
# RandomForest derived feature + Raw feature +LR


X_train_ext = hstack([one_hot.transform(rf.apply(X_train)), X_train])
X_test_ext = hstack([one_hot.transform(rf.apply(X_test)), X_test])
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train_ext,y_train)
y_pred = lr.predict_proba(X_test_ext)[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for RandomForest derived feature + Raw feature + LR：", roc_auc_score(y_test, y_pred))



# ##  Feature Learning by Trees
# Use only Raw Feature + LR


lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train,y_train)
y_pred = lr.predict_proba(X_test)[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for RandomForest derived feature + LR：", roc_auc_score(y_test, y_pred))



# ## Feature Learning by Trees
#
# Use only Raw Feature + GBDT


gbdt = GradientBoostingClassifier(n_estimators=20)

X_train = X_train[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)

gbdt.fit(X_train, y_train)
y_pred = gbdt.predict_proba(X_test)[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for Raw feature + GBDT：", roc_auc_score(y_test, y_pred))



# ## Feature Learning by Trees
#
# Use only Raw Feature + RF


rf = RandomForestClassifier(n_estimators=20)

X_train = X_train[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)
X_test = X_test[[ 'Pclass', 'Age', 'Fare', 'SibSp']].fillna(0)

rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:,1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred)
print("AUC for Raw feature + RF：", roc_auc_score(y_test, y_pred))


# #### Without tuning, we can see GBDT derived feature + LR get the best result

