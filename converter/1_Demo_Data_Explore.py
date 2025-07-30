# Generated from: 1_Demo_Data_Explore.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import os

sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')


%matplotlib inline
from data_exploration import explore


# ## Read the dataset


# temp = pd.read_csv('./data/titanic.csv')
temp = pd.read_csv("/home/hung/Data/fxsb/USDxxx/GBPUSD/GBPUSD_240.csv")
temp.head()


temp['time'] = pd.to_datetime(temp['time'])


temp.info()


use_cols = ['time', 'close', 'volume']
data = pd.read_csv('/home/hung/Data/fxsb/USDxxx/GBPUSD/GBPUSD_240.csv', usecols=use_cols, parse_dates=['time'])


data['change'] = data['close'].pct_change()*100


data.dropna(inplace=True)


data.info()


data.head()


# ## Get dtypes for each columns


str_var_list, num_var_list, all_var_list = explore.get_dtypes(data=data)


print(str_var_list) # string type
print(num_var_list) # numeric type
print(all_var_list) # all


# ## General data description


explore.describe(data=data,output_path=r'./output/')


# ## Discrete variable barplot
# draw the barplot of a discrete variable x against y(target variable). 
# By default the bar shows the mean value of y.


explore.discrete_var_barplot(x='close',y='volume',data=data,output_path='./output/')


# ## Discrete variable countplot
# draw the countplot of a discrete variable x


explore.discrete_var_countplot(x='Pclass',data=data,output_path='./output/')


# ## Discrete variable boxplot
# draw the boxplot of a discrete variable x against y.


explore.discrete_var_boxplot(x='Pclass',y='Fare',data=data,output_path='./output/')


# ## Continuous variable distplot
# draw the distplot of a continuous variable x.


explore.continuous_var_distplot(x=data['Fare'],output_path='./output/')


# ## Scatter plot
# draw the scatter-plot of two variables.


explore.scatter_plot(x=data.Fare,y=data.Pclass,data=data,output_path='./output/')


# ## Correlation plot
# draw the correlation plot between variables.


data['Sex'].value_counts().plot(kind='bar',color='c',rot=0)


data["Sex"].info()


data["Sex"] = data["Sex"].astype('category')
data["Sex"].info()


data["Sex"] = data["Sex"].cat.codes
data


explore.correlation_plot(data=data,output_path='./output/')


# ## Heatmap


flights = sns.load_dataset("flights")
print(flights.head(5))
# explore.heatmap(data=data[['Sex','Survived']])
flights = flights.pivot(index="month", columns="year", values="passengers")
# flights = flights.pivot_table("month", "year", "passengers")
explore.heatmap(data=flights,output_path='./output/')

