import pandas as pd
import numpy as np
import os
from feature_cleaning import outlier as ot
use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
print(data.shape)
pd.Series(data.Fare.unique()).sort_values()
index, para = ot.outlier_detect_arbitrary(data=data, col='Fare',
    upper_fence=300, lower_fence=5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data.loc[index, 'Fare'].sort_values()
index, para = ot.outlier_detect_IQR(data=data, col='Fare', threshold=5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data.loc[index, 'Fare'].sort_values()
index, para = ot.outlier_detect_mean_std(data=data, col='Fare', threshold=3)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data.loc[index, 'Fare'].sort_values()
index = ot.outlier_detect_MAD(data=data, col='Fare', threshold=3.5)
index, para = ot.outlier_detect_arbitrary(data=data, col='Fare',
    upper_fence=300, lower_fence=5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data[255:275]
data2 = ot.impute_outlier_with_arbitrary(data=data, outlier_index=index,
    value=-999, col=['Fare'])
data2[255:275]
index, para = ot.outlier_detect_arbitrary(data, 'Fare', 300, 5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data3 = ot.windsorization(data=data, col='Fare', para=para, strategy='both')
data3[255:275]
index, para = ot.outlier_detect_arbitrary(data, 'Fare', 300, 5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data4 = ot.drop_outlier(data=data, outlier_index=index)
print(data4.Fare.max())
print(data4.Fare.min())
index, para = ot.outlier_detect_arbitrary(data, 'Fare', 300, 5)
print('Upper bound:', para[0], '\nLower bound:', para[1])
data5 = ot.impute_outlier_with_avg(data=data, col='Fare', outlier_index=
    index, strategy='mean')
data5[255:275]