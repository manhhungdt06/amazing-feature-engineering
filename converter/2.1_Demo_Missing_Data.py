# Generated from: 2.1_Demo_Missing_Data.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')


%matplotlib inline
from feature_cleaning import missing_data as ms


# ## Load dataset


use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]

data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
print(data.shape)
data.head(8)


# ## Missing value checking
# check the total number & percentage of missing values
# per variable of a pandas Dataframe


# only variable Age has missing values, totally 177 cases
# result is saved at the output dir (if given)

ms.check_missing(data=data,output_path=r'./output/')


# ## Listwise deletion  
# excluding all cases (listwise) that have missing values


# 177 cases which has NA has been dropped 
data2 = ms.drop_missing(data=data)
data2.shape


# ## Add a variable to denote NA
# creating an additional variable indicating whether the data was missing for that observation


# Age_is_NA is created, 0-not missing 1-missing for that observation
data3 = ms.add_var_denote_NA(data=data,NA_col=['Age'])
print(data3.Age_is_NA.value_counts())
data3.head(8)


# ## Arbitrary Value Imputation
# Replacing the NA by arbitrary values


data4 = ms.impute_NA_with_arbitrary(data=data,impute_value=-999,NA_col=['Age'])
data4.head(8)


# ## Mean/Median/Mode Imputation
# Replacing the NA by mean/median/mode of that variable


print(data.Age.median())
data5 = ms.impute_NA_with_avg(data=data,strategy='median',NA_col=['Age'])
data5.head(8)


# ##  End of distribution Imputation
# replacing the NA by values that are at the far end of the distribution of that variable
# calculated by mean + 3*std


data6 = ms.impute_NA_with_end_of_distribution(data=data,NA_col=['Age'])
data6.head(8)


# ##  Random Imputation
# replacing the NA with random sampling from the pool of available observations of the variable


data7 = ms.impute_NA_with_random(data=data,NA_col=['Age'])
data7.head(8)

