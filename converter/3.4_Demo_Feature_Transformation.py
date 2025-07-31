import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from feature_engineering import transformation
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
    test_size=0.3, random_state=0)
X_train.shape, X_test.shape
X_train_copy = X_train.copy(deep=True)
X_train_copy = transformation.log_transform(data=X_train, cols=['Fare'])
print(X_train_copy.head(6))
X_train_copy = X_train.copy(deep=True)
X_train_copy = X_train_copy[X_train_copy.Fare != 0]
X_train_copy = transformation.reciprocal_transform(data=X_train_copy, cols=
    ['Fare'])
print(X_train_copy.head(6))
X_train_copy = X_train.copy(deep=True)
X_train_copy = transformation.square_root_transform(data=X_train, cols=['Fare']
    )
print(X_train_copy.head(6))
X_train_copy = X_train.copy(deep=True)
X_train_copy = transformation.exp_transform(data=X_train, cols=['Fare'],
    coef=0.2)
print(X_train_copy.head(6))
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer().fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_boxcox'] = pt.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))
transformation.diagnostic_plots(X_train_copy, 'Fare_boxcox')
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal').fit(X_train[['Fare']])
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_qt'] = qt.transform(X_train_copy[['Fare']])
print(X_train_copy.head(6))
transformation.diagnostic_plots(X_train_copy, 'Fare_qt')