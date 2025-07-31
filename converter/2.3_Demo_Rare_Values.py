import pandas as pd
import numpy as np
import os
from feature_cleaning import rare_values as ra
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
for i in ['Pclass', 'SibSp']:
    print('Variable', i, 'label proportion:')
    print(data[i].value_counts() / len(data))
enc = ra.GroupingRareValues(cols=['Pclass', 'SibSp'], threshold=0.01).fit(data)
print(enc.mapping)
data2 = enc.transform(data)
print(data2.SibSp.value_counts())
enc = ra.ModeImputation(cols=['Pclass', 'SibSp'], threshold=0.01).fit(data)
print(enc.mapping)
data3 = enc.transform(data)
print(data3.SibSp.value_counts())