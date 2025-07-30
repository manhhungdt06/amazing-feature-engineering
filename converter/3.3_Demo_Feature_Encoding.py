# Generated from: 3.3_Demo_Feature_Encoding.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

import category_encoders as ce
from feature_engineering import encoding



# ## Load Dataset


use_cols = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp',
    'Survived'
]

data = pd.read_csv('./data/titanic.csv', usecols=use_cols)
data.head()


X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# ## One-hot encoding
# replace the categorical variable by different boolean variables (0/1) to indicate whether or not certain label is true for that observation


data1 = pd.get_dummies(data,drop_first=True)


data1.head()


# ## Ordinal-encoding
# replace the labels by some ordinal number if ordinal is meaningful


ord_enc = ce.OrdinalEncoder(cols=['Sex']).fit(X_train,y_train)


data4 = ord_enc.transform(data)
print(data4.head(5))


# ## Mean encoding
# replace the label by the mean of the target for that label. 
# (the target must be 0/1 valued or continuous)


# cross check-- the mean of target group by Sex
X_train['Survived'].groupby(data['Sex']).mean()



mean_enc = encoding.MeanEncoding(cols=['Sex']).fit(X_train,y_train)


data6 = mean_enc.transform(data)
print(data6.head(5))


# ## Target-encoding
# Similar to mean encoding, but use both posterior probability and prior probability of the target


# create the encoder and fit with our data
target_enc = ce.TargetEncoder(cols=['Sex']).fit(X_train,y_train)


# perform transformation
# data.Survived.groupby(data['Sex']).agg(['mean'])
data2 = target_enc.transform(data)


# check the result
data2.head()


# ## WOE-encoding
# replace the label  with Weight of Evidence of each label. WOE is computed from the basic odds ratio: 
#
# ln( (Proportion of Good Outcomes) / (Proportion of Bad Outcomes))


woe_enc = ce.WOEEncoder(cols=['Sex']).fit(X_train,y_train)


data3 = woe_enc.transform(data)


data3.head(5)

