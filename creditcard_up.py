# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:14:32 2020

@author: user
"""

#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# ### Presented by Eduonix!
# 
# Throughout the financial sector, machine learning algorithms are being developed to detect fraudulent transactions.  In this project, that is exactly what we are going to be doing as well.  Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, we are going to identify transactions with a high probability of being credit card fraud.  In this project, we will build and deploy the following two machine learning algorithms:
# 
# * Local Outlier Factor (LOF)
# * Isolation Forest Algorithm
# 
# Furthermore, using metrics suchs as precision, recall, and F1-scores, we will investigate why the classification accuracy for these algorithms can be misleading.
# 
# In addition, we will explore the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in our data set. Let's get started!
# 
# ## 1. Importing Necessary Libraries
# 
# To start, let's print out the version numbers of all the libraries we will be using in this project. This serves two purposes - it ensures we have installed the libraries correctly and ensures that this tutorial will be reproducible. 

# In[1]:
from IPython import get_ipython 
get_ipython().magic('clear')
import numpy
import pandas
import seaborn
import sklearn.model_selection as ms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nr
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# In[2]:


# import the necessary packages




# ### 2. The Data Set
# 
# In the following cells, we will import our dataset from a .csv file as a Pandas DataFrame.  Furthermore, we will begin exploring the dataset to gain an understanding of the type, quantity, and distribution of data in our dataset.  For this purpose, we will use Pandas' built-in describe feature, as well as parameter histograms and a correlation matrix. 

# In[3]:


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')
data_fraud=data[data.Class==1]
data_normal=data[data.Class==0].sample(n=len(data_fraud))
# In[4]:


# Start exploring the dataset
print(data.columns)


# In[5]:


# Print the shape of the data
#data = data.sample(frac=0.4, random_state = 1)
data=pd.concat([data_fraud,data_normal])
print(data.shape)
print(data.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features


# In[6]:


# Plot histograms of each parameter 
data.hist(figsize = (10, 10))
plt.show()


# In[7]:


# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
outlier_fraction=0.001
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

"""
# In[8]:
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")

# In[8]:
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] == 0.232013) ]
data1=data.drop(data[to_drop], axis=1)
a=get_top_abs_correlations(corr_matrix, 20*20)

"""
# In[8]:
# Correlation matrix
corrmat = data.corr().abs()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[8]:
best_correlation=corrmat.Class[corrmat.Class>0.5]
data=data[best_correlation.index]
# In[9]:


# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)



X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=42)



## Define the dictionary for the grid search and the model object to search on
param_grid = {"max_features": [2, 3, 5, 8], "min_samples_leaf":[3, 5, 10, 20]}
## Define the random forest model
nr.seed(123)
rf_clf = RandomForestClassifier(class_weight = "balanced") # class_weight = {0:0.33, 1:0.67}) 

## Perform the grid search over the parameters
nr.seed(123)
rf_clf = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid, 
                      scoring = 'roc_auc',
                      return_train_score = True)
rf_clf.fit(X_train, Y_train)
print(rf_clf.best_estimator_.max_features)
print(rf_clf.best_estimator_.min_samples_leaf)





nr.seed(123)
rf_clf2=RandomForestClassifier( min_samples_leaf=rf_clf.best_estimator_.min_samples_leaf,max_features=rf_clf.best_estimator_.max_features)
cv_estimate = ms.cross_val_score(rf_clf2, X_train, Y_train) # Use the outside folds
# cv_estimate_2 = ms.cross_validate(clf, label_train, label_train, cv = outside) # Use the outside folds
print('Mean performance metric = %4.3f' % np.mean(cv_estimate))

print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))



rf_clf=RandomForestClassifier( min_samples_leaf=rf_clf.best_estimator_.min_samples_leaf,max_features=rf_clf.best_estimator_.max_features)
rf_clf.fit(X_train, Y_train) 
ko=rf_clf.predict(X_test)
metrics = sklm.precision_recall_fscore_support(Y_test.values, ko)
conf = sklm.confusion_matrix(Y_test.values, ko)
print('                 Confusion matrix')
print('                 Score positive    Score negative')
print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
print('')
print('Accuracy  %0.2f' % sklm.accuracy_score(Y_test.values, ko))
print(' ')
print('           Positive      Negative')
print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])





## Define the dictionary for the grid search and the model object to search on
param_grid = {"learning_rate": [0.1, 1, 10]}
## Define the AdaBoosted tree model
nr.seed(123)
ab_clf = AdaBoostClassifier()  

## Perform the grid search over the parameters
nr.seed(123)
ab_clf = ms.GridSearchCV(estimator = ab_clf, param_grid = param_grid, 
                      scoring = 'roc_auc',
                      return_train_score = True)
ab_clf.fit(X_train, Y_train)
print(ab_clf.best_estimator_.learning_rate)


nr.seed(123)
ada_clf2 = AdaBoostClassifier(learning_rate=ab_clf.best_estimator_.learning_rate, n_estimators=ab_clf.best_estimator_.n_estimators)
cv_estimate = ms.cross_val_score(ada_clf2, X_train, Y_train) # Use the outside folds
# cv_estimate_2 = ms.cross_validate(clf, label_train, label_train, cv = outside) # Use the outside folds
print('Mean performance metric = %4.3f' % np.mean(cv_estimate))

print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate):
    print('Fold %2d    %4.3f' % (i+1, x))
    
    
ada_clf = AdaBoostClassifier(learning_rate=ab_clf.best_estimator_.learning_rate, n_estimators=ab_clf.best_estimator_.n_estimators)
ada_clf.fit(X_train, Y_train) 
ko=ada_clf.predict(X_test)
metrics = sklm.precision_recall_fscore_support(Y_test.values, ko)
conf = sklm.confusion_matrix(Y_test.values, ko)
print('                 Confusion matrix')
print('                 Score positive    Score negative')
print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
print('')
print('Accuracy  %0.2f' % sklm.accuracy_score(Y_test.values, ko))
print(' ')
print('           Positive      Negative')
print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])




# ## 3. Unsupervised Outlier Detection
# 
# Now that we have processed our data, we can begin deploying our machine learning algorithms.  We will use the following techniques: 
# 
# **Local Outlier Factor (LOF)**
# 
# The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a 
# given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the 
# object is with respect to the surrounding neighborhood.
# 
# 
# **Isolation Forest Algorithm**
# 
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting 
# a split value between the maximum and minimum values of the selected feature.
# 
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to 
# isolate a sample is equivalent to the path length from the root node to the terminating node.
# 
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# 
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees 
# collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

# In[11]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}


# In[15]:


# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

