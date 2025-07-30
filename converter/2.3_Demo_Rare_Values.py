# Generated from: 2.3_Demo_Rare_Values.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
# plt.style.use('seaborn-colorblind')
# %matplotlib inline
from feature_cleaning import rare_values as ra


# ## Load Dataset


use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]

# see column Pclass & SibSp's distributions
# SibSp has values 3/8/5 that occur rarely, under 2%
# Pclass has 3 values, but no one is under 20%
data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
for i in ['Pclass','SibSp']:
    print('Variable',i,'label proportion:')
    print(data[i].value_counts()/len(data))


# ## Grouping into one new category
# Grouping the observations that show rare labels into a unique category ('rare')


# create the encoder and fit with our data
enc = ra.GroupingRareValues(cols=['Pclass','SibSp'],threshold=0.01).fit(data)


# let's see the mapping
# for SibSp, values 5 & 8 are encoded as 'rare' as they appear less than 10%
# for Pclass, nothing changed
print(enc.mapping)


# perform transformation
data2 = enc.transform(data)


# check the result
print(data2.SibSp.value_counts())


# ## Mode Imputation
# Replacing the rare label by most frequent label


# create the encoder and fit with our data
enc = ra.ModeImputation(cols=['Pclass','SibSp'],threshold=0.01).fit(data)


# let's see the mapping
# for SibSp, values 5 & 8 are encoded as 0, as label 0 is the most frequent label
# for Pclass, nothing changed
print(enc.mapping)


# perform transformation
data3 = enc.transform(data)


# check the result
print(data3.SibSp.value_counts())

