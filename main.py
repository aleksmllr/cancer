
# coding: utf-8

# In[447]:

# Importing useful packages to make life easy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold


# In[448]:

# Load data
df1 = pd.read_csv('data/data_set_ALL_AML_independent.csv')
df2 = pd.read_csv('data/data_set_ALL_AML_train.csv')
df3 = pd.read_csv('data/actual.csv')

df1.head()


# In[449]:

# get columns to drop
df1_drop = [column for column in df1.columns if 'call' in column]
df2_drop = [column for column in df2.columns if 'call' in column]


# In[450]:

# drop columns
df1 = df1.drop(df1_drop, axis=1)
df2 = df2.drop(df2_drop, axis=1)


# In[451]:

df1.head()


# In[452]:

# transpose the frames so expression levels of each gene are columns and patients are rows
df1 = df1.T
df2 = df2.T


# In[453]:

df1.head()


# In[454]:

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
X_train = train.iloc[:, 1:].values.astype('float64')

y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.astype('float64')

y_test = test.iloc[:, 0].values


# To assess principle components data must be standardized
# Instantiate standard scaler and PCA
sc = StandardScaler()



X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)


# In[459]:

# Remove columns with low variance
selector = VarianceThreshold(threshold=0.3)

reduced = selector.fit_transform(X_train)

print(reduced.shape, X_train.shape)


# In[460]:

# Run principle component analysis to retain 99% of the variance present in the data
pca = PCA(n_components=.99)

pComponents = pca.fit_transform(X_train_std)

#pca.explained_variance_ratio_.shape

var_exp = pca.explained_variance_ratio_

cum_var_exp = np.cumsum(var_exp)


# In[395]:

pComponents.shape


# In[396]:

plt.bar(range(1,37), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 37), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()


# In[397]:

cum_var_exp[31]


# In[398]:

#With 30 components we retain about 95% of the variance
pca = PCA(n_components=30)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# In[399]:

X_train_pca.shape


# In[400]:

X_test_pca.shape


# In[407]:

X = np.concatenate((X_train_pca, X_test_pca))


# In[408]:

y = np.concatenate((y_train, y_test))


# In[419]:

## Now I am going to use GridSearch on SVM, Knn, and RandomForest
# Import
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# In[420]:

svc_pipe = make_pipeline(StandardScaler(), PCA(n_components=30), 
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range,
              'svc__kernel': ['linear']},
             {'svc__C': param_range,
             'svc__gamma': param_range,
             'svc__kernel': ['rbf']}]
gs1 = GridSearchCV(estimator=svc_pipe,
                  param_grid=param_grid,
                  scoring= None,
                  cv=10,
                  n_jobs=-1)

gs1 = gs1.fit(X_train, y_train)


# In[421]:

best_svm = gs1.best_estimator_

best_svm.fit(X_train_pca, y_train)

y_pred = best_svm.predict(X_test_pca)

print("The best training score:", gs1.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# In[422]:

knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [i for i in range(1,25,5)],
             'weights': ['uniform', 'distance'],
             'algorithm': ["ball_tree", "kd_tree", "brute"],
             "leaf_size": [1, 10, 30], 
              "p": [1,2]}

gs2 = GridSearchCV(estimator=knn, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True)
                         
gs2.fit(X_train_pca, y_train)


# In[424]:

best_knn = gs2.best_estimator_

best_knn.fit(X_train_pca, y_train)

y_pred = best_knn.predict(X_test_pca)

print("The best training score:", gs2.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# In[430]:

rf = RandomForestClassifier()

param_grid = {'max_depth': [None],
             'min_samples_split': [2,3],
             'min_samples_leaf': [1],
             'min_weight_fraction_leaf': [0.], 
              'max_features': [None],
             'random_state': [3],
             'max_leaf_nodes': [None]}

gs3 = GridSearchCV(estimator=rf, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True)
                         
gs3.fit(X_train_pca, y_train)


# In[432]:

best_rf = gs3.best_estimator_

best_rf.fit(X_train_pca, y_train)

y_pred = best_knn.predict(X_test_pca)

print("The best training score:", gs3.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:



