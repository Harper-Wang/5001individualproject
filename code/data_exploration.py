"""
Data exploration
"""

#%%
# load packages
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
traindata = pd.read_csv('/Users/wangchuhan/Desktop/kaggle_github/data/train.csv')
testdata = pd.read_csv('/Users/wangchuhan/Desktop/kaggle_github/data/test.csv')

# have a quick glance at dataframe structure
display(traindata.head())
display(testdata.head())


#%%
numerical_columns = ['l1_ratio', 'alpha', 'max_iter', 'random_state'
, 'n_jobs', 'n_samples', 'n_features','n_classes', 'n_clusters_per_class'
, 'n_informative', 'flip_y', 'scale']

categorical_columns = ['penalty']

target = 'time'
#%%
# do visualization to find correlation between attributes and target
def myplot(df, numerical_columns, categorical_columns, target):  
        for col in numerical_columns:  
                plt.figure(figsize=(8,6))
                plt.scatter(df[col].values,df[target].values)
                plt.xlabel(col)
                plt.ylabel(target)
                plt.title(col+' vs '+target)
                plt.savefig(os.path.join('.../graphs/',col+'_vs_'+target+'.png')) # change path to your own
                plt.show()
               
        for col in categorical_columns:
                dict1 = dict(traindata.groupby(['penalty'])[target].mean())
                x = list(dict1.keys())
                height = list(dict1.values())
                plt.figure(figsize=(8,6))
                plt.bar(x, height)
                plt.xlabel(col)
                plt.ylabel(target)
                plt.title(col+' vs '+target)
                plt.savefig(os.path.join('.../graphs/',col+'_vs_'+target+'.png')) # change path to your own
                plt.show()

myplot(traindata,numerical_columns, categorical_columns,target)