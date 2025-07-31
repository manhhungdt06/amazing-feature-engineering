import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import category_encoders as ce
from feature_engineering import encoding
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head()
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
    test_size=0.3, random_state=0)
X_train.shape, X_test.shape
data1 = pd.get_dummies(data, drop_first=True)
data1.head()
ord_enc = ce.OrdinalEncoder(cols=['Sex']).fit(X_train, y_train)
data4 = ord_enc.transform(data)
print(data4.head(5))
X_train['Survived'].groupby(data['Sex']).mean()
mean_enc = encoding.MeanEncoding(cols=['Sex']).fit(X_train, y_train)
data6 = mean_enc.transform(data)
print(data6.head(5))
target_enc = ce.TargetEncoder(cols=['Sex']).fit(X_train, y_train)
data2 = target_enc.transform(data)
data2.head()
woe_enc = ce.WOEEncoder(cols=['Sex']).fit(X_train, y_train)
data3 = woe_enc.transform(data)
data3.head(5)