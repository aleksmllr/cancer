# testing first commit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df1 = pd.read_csv('data/data_set_ALL_AML_independent.csv')
df2 = pd.read_csv('data/data_set_ALL_AML_train.csv')
df3 = pd.read_csv('data/actual.csv')
# get columns to drop
df1_drop = [column for column in df1.columns if 'call' in column]
df2_drop = [column for column in df2.columns if 'call' in column]
# drop columns
df1 = df1.drop(df1_drop, axis=1)
df2 = df2.drop(df2_drop, axis=1)

df1.head()
df2.head()
# transpose the frames so expression levels of each gene are columns and patients are rows
df1 = df1.T
df2 = df2.T

df1.head()
df2.head()
#Set the columns names to the Gene Accession Number
df1.columns = df1.iloc[1]
df2.columns = df2.iloc[1]
# Drop remaining columns and convert to int64
df1 = df1.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)
df2 = df2.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

#df2.iloc[0][1]

# Reset indexes for concatenation with labels

df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

# Isolate labels for training set
labels_train = df3.iloc[:38, 1].reset_index(drop=True)
# Concatenate the two frames on the columns (axis=1) to form the training set
train = pd.concat([labels_train, df2], axis=1)

# Isolate labels for test set
labels_test = df3.iloc[38:, 1].reset_index(drop=True)

# Concatenate the two frames on the columns (axis=1) to form the test set
test = pd.concat([labels_test, df1], axis=1)

# Prepare train and test sets
X_train = train.iloc[:, 1:].values

y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values

y_test = test.iloc[:, 0].values


# To assess principle components data must be standardized
# Instantiate standard scaler and PCA
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

X_train.shape


# initiate PCA object for all components to assess which components explain the most variance
pca = PCA(n_components=30)

pComponents = pca.fit_transform(X_train)

pca.explained_variance_ratio_.shape

var_exp = pca.explained_variance_ratio_

cum_var_exp = np.cumsum(var_exp)










# Take a random sample of the features with out replacement
#sample = train.iloc[:, 1:].sample(n=250, replace=False, axis=1)
#sample.describe().round()
