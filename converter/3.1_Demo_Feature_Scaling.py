# Generated from: 3.1_Demo_Feature_Scaling.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

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


# ## Normalization - Standardization (Z-score scaling)
#
# removes the mean and scales the data to unit variance.<br />z = (X - X.mean) /  std


# add the new created feature
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_zscore'] = ss.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))


# check if it is with mean=0 std=1
print(X_train_copy['Fare_zscore'].mean())
print(X_train_copy['Fare_zscore'].std())



# ## Min-Max scaling
# transforms features by scaling each feature to a given range. Default to [0,1].<br />X_scaled = (X - X.min / (X.max - X.min)


# add the new created feature
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_minmax'] = mms.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))


# check the range of Fare_minmax
print(X_train_copy['Fare_minmax'].max())
print(X_train_copy['Fare_minmax'].min())


# ## Robust scaling
# removes the median and scales the data according to the quantile range (defaults to IQR)<br />X_scaled = (X - X.median) / IQR


# add the new created feature
from sklearn.preprocessing import RobustScaler
rs = RobustScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_robust'] = rs.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))

