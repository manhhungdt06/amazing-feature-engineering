import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')
%matplotlib inline
from feature_cleaning import missing_data as ms
use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
print(data.shape)
data.head(8)
ms.check_missing(data=data,output_path=r'./output/')
data2 = ms.drop_missing(data=data)
data2.shape
data3 = ms.add_var_denote_NA(data=data,NA_col=['Age'])
print(data3.Age_is_NA.value_counts())
data3.head(8)
data4 = ms.impute_NA_with_arbitrary(data=data,impute_value=-999,NA_col=['Age'])
data4.head(8)
print(data.Age.median())
data5 = ms.impute_NA_with_avg(data=data,strategy='median',NA_col=['Age'])
data5.head(8)
data6 = ms.impute_NA_with_end_of_distribution(data=data,NA_col=['Age'])
data6.head(8)
data7 = ms.impute_NA_with_random(data=data,NA_col=['Age'])
data7.head(8)