import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
    test_size=0.3, random_state=0)
X_train.shape, X_test.shape
from sklearn.preprocessing import StandardScaler
ss = StandardScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_zscore'] = ss.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))
print(X_train_copy['Fare_zscore'].mean())
print(X_train_copy['Fare_zscore'].std())
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_minmax'] = mms.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))
print(X_train_copy['Fare_minmax'].max())
print(X_train_copy['Fare_minmax'].min())
from sklearn.preprocessing import RobustScaler
rs = RobustScaler().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_robust'] = rs.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))