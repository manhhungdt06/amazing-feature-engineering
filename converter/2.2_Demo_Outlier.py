# Generated from: 2.2_Demo_Outlier.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from feature_cleaning import outlier as ot


# ## Load dataset


use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]


data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head(3)
print(data.shape)


pd.Series(data.Fare.unique()).sort_values()


# ## Detect by arbitrary boundary
# identify outliers based on arbitrary boundaries


index,para = ot.outlier_detect_arbitrary(data=data,col='Fare',upper_fence=300,lower_fence=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# check the 19 found outliers
data.loc[index,'Fare'].sort_values()


# ## IQR method
# outlier detection by Interquartile Ranges Rule


index,para = ot.outlier_detect_IQR(data=data,col='Fare',threshold=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# check the 31 found outliers
data.loc[index,'Fare'].sort_values()


# ## Mean and Standard Deviation Method
# outlier detection by Mean and Standard Deviation Method.


index,para = ot.outlier_detect_mean_std(data=data,col='Fare',threshold=3)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# check the 20 found outliers
data.loc[index,'Fare'].sort_values()


# ## MAD method
# outlier detection by Median and Median Absolute Deviation Method (MAD)


# too aggressive for our dataset, about 18% of cases are detected as outliers.
index = ot.outlier_detect_MAD(data=data,col='Fare',threshold=3.5)


# ##  Imputation with arbitrary value
# impute outliers with arbitrary value


# use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data=data,col='Fare',upper_fence=300,lower_fence=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


data[255:275]


# see index 258,263,271 have been replaced
data2 = ot.impute_outlier_with_arbitrary(data=data,outlier_index=index,
                                         value=-999,col=['Fare'])
data2[255:275]


# ## Windsorization
# top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value,vice versa)


# use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data,'Fare',300,5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# see index 258,263,271 have been replaced with top/bottom coding

data3 = ot.windsorization(data=data,col='Fare',para=para,strategy='both')
data3[255:275]


# ## Discard outliers
# Drop the cases that are outliers


# use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data,'Fare',300,5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# drop the outlier.
# we can see no more observations have value >300 or <5. They've been removed.
data4 = ot.drop_outlier(data=data,outlier_index=index)
print(data4.Fare.max())
print(data4.Fare.min())


# ## Mean/Median/Mode Imputation
# replacing the outlier by mean/median/most frequent values of that variable


# use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data,'Fare',300,5)
print('Upper bound:',para[0],'\nLower bound:',para[1])
    


# see index 258,263,271 have been replaced with mean

data5 = ot.impute_outlier_with_avg(data=data,col='Fare',
                                   outlier_index=index,strategy='mean')
data5[255:275]

