import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from feature_engineering import discretization as dc
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
X_train, X_test, y_train, y_test = train_test_split(data, data.Survived,
    test_size=0.3, random_state=0)
X_train.shape, X_test.shape
from sklearn.preprocessing import KBinsDiscretizer
enc_equal_width = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=
    'uniform').fit(X_train[['Fare']])
enc_equal_width.bin_edges_
result = enc_equal_width.transform(X_train[['Fare']])
pd.DataFrame(result)[0].value_counts()
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_equal_width'] = enc_equal_width.transform(X_train[['Fare']])
print(X_train_copy.head(10))
enc_equal_freq = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=
    'quantile').fit(X_train[['Fare']])
enc_equal_freq.bin_edges_
result = enc_equal_freq.transform(X_train[['Fare']])
pd.DataFrame(result)[0].value_counts()
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_equal_freq'] = enc_equal_freq.transform(X_train[['Fare']])
print(X_train_copy.head(10))
enc_kmeans = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans'
    ).fit(X_train[['Fare']])
enc_kmeans.bin_edges_
result = enc_kmeans.transform(X_train[['Fare']])
pd.DataFrame(result)[0].value_counts()
X_train_copy = X_train.copy(deep=True)
X_train_copy['Fare_kmeans'] = enc_kmeans.transform(X_train[['Fare']])
print(X_train_copy.head(10))
enc1 = dc.DiscretizeByDecisionTree(col='Fare', max_depth=2).fit(X=X_train,
    y=y_train)
enc1.tree_model
data1 = enc1.transform(data)
print(data1.head(5))
print(data1.Fare_tree_discret.unique())
col = 'Fare'
bins = pd.concat([data1.groupby([col + '_tree_discret'])[col].min(), data1.
    groupby([col + '_tree_discret'])[col].max()], axis=1)
print(bins)
enc2 = dc.DiscretizeByDecisionTree(col='Fare', max_depth=[2, 3, 4, 5, 6, 7]
    ).fit(X=X_train, y=y_train)
enc2.tree_model
data2 = enc2.transform(data)
data2.head(5)
enc3 = dc.ChiMerge(col='Fare', num_of_bins=5).fit(X=X_train, y='Survived')
enc3.bins
data3 = enc3.transform(data)
print(data3.head(5))
data3.Fare_chimerge.unique()