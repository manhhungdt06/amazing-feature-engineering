# Generated from: 4.5_Demo_Feature_Selection_Hybrid_method.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from sklearn.feature_selection import RFE
from feature_selection import hybrid



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


# ##  Recursive Feature Elimination 
# ### with Random Forests Importance


# ### Example 1
# This method is slightly **different from the guide**, as it use a different stopping criterion: the desired number of features to select is eventually reached.


#  n_features_to_select decide the stopping criterion
# we stop till 10 features remaining

sel_ = RFE(RandomForestClassifier(n_estimators=20), n_features_to_select=10)
sel_.fit(X_train.fillna(0), y_train)


selected_feat = X_train.columns[(sel_.get_support())]
print(selected_feat)


# ### Example 2
# recursive feature elimination with RandomForest
# with the method same as the guide
# 1. Rank the features according to their importance derived from a machine learning algorithm: it can be tree importance, or LASSO / Ridge,  or the linear / logistic regression coefficients.
# 2. Remove one feature -the least important- and build a machine learning algorithm utilizing the remaining features.
# 3. Calculate a performance metric of your choice: roc-auc, mse, rmse, accuracy.
# 4. If the metric decreases by more of an arbitrarily set threshold, then that feature is important and should be kept. Otherwise, we can remove that feature.
# 5. Repeat steps 2-4 until all features have been removed (and therefore evaluated) and the drop in performance assessed.


# tol decide whether we should drop or keep the feature in current round
features_to_keep = hybrid.recursive_feature_elimination_rf(X_train=X_train,
                                                           y_train=y_train,
                                                           X_test=X_test,
                                                           y_test=y_test,
                                                           tol=0.001)


features_to_keep


# ## Recursive Feature Addition
# ### with Random Forests Importance


# ### Example 1
# recursive feature addition with RandomForest
# with the method same as the guide
# 1. Rank the features according to their importance derived from a  machine learning algorithm: it can be tree importance, or LASSO / Ridge,  or the linear / logistic regression coefficients.
# 2. Build a machine learning model with only 1 feature, the most important one, and calculate the model metric for performance.
# 3. Add one feature -the most important- and build a machine learning  algorithm utilizing the added and any feature from previous rounds.
# 4. Calculate a performance metric of your choice: roc-auc, mse, rmse, accuracy.
# 5. If the metric increases by more than an arbitrarily set threshold,  then that feature is important and should be kept. Otherwise, we can  remove that feature.
# 6. Repeat steps 2-5 until all features have been removed (and therefore evaluated) and the drop in performance assessed.


features_to_keep = hybrid.recursive_feature_addition_rf(X_train=X_train,
                                                        y_train=y_train,
                                                        X_test=X_test,
                                                        y_test=y_test,
                                                        tol=0.001)


features_to_keep

