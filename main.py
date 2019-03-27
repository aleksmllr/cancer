# testing first commit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('data/data_set_ALL_AML_independent.csv')
df2 = pd.read_csv('data/data_set_ALL_AML_train.csv')
df3 = pd.read_csv('data/actual.csv')
#df1.head()

#df1.shape
#df2.shape
#df3.shape

#df1 = df1.T
#df2 = df2.T

df1.shape
df2.shape
df3.shape


df1 = df1.drop([column for column in df1.columns if 'call' in column], axis=1).T
df2 = df2.drop([column for column in df2.columns if 'call' in column], axis=1).T

df2.head(15)
