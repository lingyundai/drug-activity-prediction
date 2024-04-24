#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
from scipy.sparse import csr_matrix
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


# import train set
# attributes are the INDICES of non-zero values
train = pd.read_csv('train.txt', header=None)


# In[3]:


train


# In[4]:


# import test set
test = pd.read_csv('test.txt', header=None)
test


# In[5]:


# split attributes and label
train = train[0].str.split('\t', expand=True)
train


# In[6]:


train_x = pd.DataFrame(train[1])
train_x


# In[7]:


train_y = pd.DataFrame(train[0])
train_y


# In[8]:


class MatrixParser:
    def __init__(self):
        pass
    
    # calculate the length needed for matrix column
    def calculateMatrixColumn(self, arr):
        colNum = 0
        for row in arr:
            for indices in row:
                # only match digits
                indices = re.findall(r'[0-9]+', indices)
                # change each index to int in np arr
                np_arr = np.array(indices, dtype=int)
            maxInd = max(np_arr)
            colNum = max(colNum, maxInd)
        print(colNum)
        return colNum
    
    # parse indices to csr matrix
    def parseIndexToMatrix(self, arr, colNum):
        res = []
        for row in arr:
            for indices in row:
                # only match digits
                indices = re.findall(r'[0-9]+', indices)
                # change each index to int in np arr
                np_arr = np.array(indices, dtype=int)
                # new arr with existing index as 1, rest is 0
                # every record has same length of arr to insert indices to
                new_arr = [0] * (colNum + 1)
                for ind in np_arr:
                    new_arr[ind] = 1
            res.append(new_arr)
            
        train_x_matrix = csr_matrix(res)
        print(train_x_matrix)
        return train_x_matrix


# In[9]:


# convert train x to csr matrix
train_x_np = train_x.to_numpy()
obj = MatrixParser()
col_num = obj.calculateMatrixColumn(train_x_np)
train_x_matrix = obj.parseIndexToMatrix(train_x_np, col_num)


# In[10]:


# convert test x to csr matrix
test_x_np = test.to_numpy()
obj = MatrixParser()
test_col_num = obj.calculateMatrixColumn(test_x_np)
test_x_matrix = obj.parseIndexToMatrix(test_x_np, test_col_num)


# In[11]:


# convert train y to numpy array
train_y = train_y.to_numpy().flatten()
train_y = train_y.astype(int)
train_y


# #### Apply mutual info classification SelectKBest
# #### Different random states for experiments for reproducibility

# In[12]:


# apply mutual information feature selection to eliminate dependent features
print("current train_x shape: ", train_x_matrix.shape)

def mutual_info(train_x_matrix, train_y):
    return mutual_info_classif(train_x_matrix, train_y, random_state=72)

mutual_info_kbest = SelectKBest(score_func=mutual_info, k=300)
train_x_matrix = mutual_info_kbest.fit_transform(train_x_matrix, train_y)
print("train_x shape after mutual info selection: ", train_x_matrix.shape)

print("current test_x shape: ", test_x_matrix.shape)
test_x_matrix = mutual_info_kbest.transform(test_x_matrix)
print("test_x shape after mutual info selection: ", test_x_matrix.shape)


# #### Apply chi2 test SelectKBest
# #### Different random states for experiments for reproducibility

# In[13]:


# # apply chi2 feature selection to eliminate irrelevant features
# print("current train_x shape: ", train_x_matrix.shape)
# chi2_kbest = SelectKBest(chi2, k=350)
# train_x_matrix = chi2_kbest.fit_transform(train_x_matrix, train_y)
# print("train_x shape after chi2 selection: ", train_x_matrix.shape)

# print("current test_x shape: ", test_x_matrix.shape)
# test_x_matrix = chi2_kbest.transform(test_x_matrix)
# print("test_x shape after chi2 selection: ", test_x_matrix.shape)


# #### Apply resampling to over sample the under represented label in train set
# #### Different random states for experiments for reproducibility

# In[14]:


# shuffle the train x and y to prevent overfitting
train_x_matrix, train_y = shuffle(train_x_matrix, train_y, random_state=72)

# produce a validation set to validate the model
train_x_matrix, validation_x, train_y, validation_y = train_test_split(
    train_x_matrix, train_y, test_size=0.15, random_state=72)


# In[15]:


# check the distribution of the labels
train_label_dist = sns.histplot(data=train_y)
train_label_dist
validation_label_dist = sns.histplot(data=validation_y)
validation_label_dist
# the labels are imbalanced


# In[16]:


# # highly imbalanced dataset, so apply resampling to over 
# # sample the under represented label in train set and validation set
# ros = RandomOverSampler(random_state=63)
# train_x_resampled, train_y_resampled = ros.fit_resample(train_x_matrix, train_y)


# In[17]:


sm = SMOTE(random_state=72)
train_x_resampled, train_y_resampled = sm.fit_resample(train_x_matrix, train_y)


# In[18]:


# check resampled dataset
sns.histplot(data=train_y_resampled)
sns.histplot(data=validation_y)

# get the amount for resampled train x validation x and test x
train_x_resampled_length = train_x_resampled.shape[0]
test_x_length = test_x_matrix.shape[0]
val_x_length = validation_x.shape[0]

# check the ratio of train val test sets after resampling
train_ratio = train_x_resampled_length / (train_x_resampled_length + test_x_length + val_x_length)
test_ratio = test_x_length / (train_x_resampled_length + test_x_length + val_x_length)
val_ratio = val_x_length / (train_x_resampled_length + test_x_length + val_x_length)

# the result shows it is roughly 70/10/20 which is common
print("train/val/test ratio: ", train_ratio, val_ratio, test_ratio)

# check the ratio of 0 and 1 in train set
# this is important because stratified cross validation fold will
# follow this ratio too
res_1 = 0
res_0 = 0
for res in train_y_resampled:
    if res == 1:
        res_1 += 1
    elif res == 0:
        res_0 += 1
# the result shows after resampling there is equal amount of 
# label 1 and label 0 in train y
print("Train label is 1: ", res_1, "Train label is 0: ", res_0)

res_1_val = 0
res_0_val = 0
for res in validation_y:
    if res == 1:
        res_1_val += 1
    elif res == 0:
        res_0_val += 1
# the result shows after resampling there is equal amount of 
# label 1 and label 0 in val y
print("Validation label is 1: ", res_1_val, "validation label is 0: ", res_0_val)


# #### Experiments for Decision Tree Classifer with Stratified Cross Validation
# #### Different random states for experiments for reproducibility, the random_state each experiment used matches random_state in RandomOverSampler above. Details can be found in the corresponding pdf file.

# In[19]:


# # shuffle the resampled data before dividing into kfolds
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=70)

# # split to n folds with the same class label ratio as dataset
# for train_index, test_index in skf.split(train_x_resampled, train_y_resampled):
#     # extract train folds x y 
#     train_x = train_x_resampled[train_index]
#     train_y = train_y_resampled[train_index]
#     # extract test fold x and y
#     test_x = train_x_resampled[test_index]
#     test_y = train_y_resampled[test_index]
    
#     # apply decision tree
#     # applying different max depth and see what works better
#     clf = tree.DecisionTreeClassifier(random_state=70)
    
#     # fit model on train folds
#     clf = clf.fit(train_x, train_y)
#     print("tree depth: ", clf.get_depth())
#     # predict on test fold
#     test_y_pred = clf.predict(test_x)
    
#     train_y_1 = 0
#     train_y_0 = 0
#     for label in train_y:
#         if label == 1:
#             train_y_1 += 1
#         elif label == 0:
#             train_y_0 += 1
    
#     test_y_1 = 0
#     test_y_0 = 0
#     for label in test_y:
#         if label == 1:
#             test_y_1 += 1
#         elif label == 0:
#             test_y_0 += 1
    
#     # train folds and test fold label ratio follows the ratio in the dataset which is good
#     print("train label ratio: ", train_y_1 / (train_y_0 + train_y_1), train_y_0 / (train_y_0 + train_y_1))
#     print("test label ratio: ", test_y_1 / (test_y_0 + test_y_1), test_y_0 / (test_y_0 + test_y_1))
    
#     f1_scores = []
#     print("f1 score: ", f1_score(test_y, test_y_pred))
#     print("confusion matrix: \n", confusion_matrix(test_y, test_y_pred))
#     f1_scores.append(f1_score(test_y, test_y_pred))

# print("average f1 score: ", sum(f1_scores) / len(f1_scores))

# # make prediction on validation set
# label_pred = clf.predict(validation_x)
# labels_pred = []
# for label in label_pred:
#     labels_pred.append(label)
# print(labels_pred)
# print(validation_y)
# print("f1 score validation set: ", f1_score(validation_y, labels_pred))
    
# visualize the tree
# decision_tree_visual = tree.export_graphviz(clf, out_file=None, filled=True) 
# graph = graphviz.Source(decision_tree_visual) 
# graph.render("decision_tree_visualization")


# In[20]:


# # predict on test set with dt clf
# label_pred = clf.predict(test_x_matrix)
# for label in label_pred:
#     print(label)


# #### The Naive Bayes algorithm that is being used here is Bernoulli Naive Bayes. The reason being in the dataset 0 and 1 are binary values that represent the absence and presence of a feature, which is suitable for Bernoulli Naive Bayes.
# 
# 
# #### Experiments for Bernoulli Naive Bayes Classifer with Stratified Cross Validation
# #### Different random states for experiments for reproducibility, the random_state each experiment used matches random_state in RandomOverSampler above. Details can be found in the corresponding pdf file.
# 

# In[21]:


# shuffle the resampled data before dividing into kfolds
skf_nb = StratifiedKFold(n_splits=5, shuffle=True, random_state=72)

# split to n folds with the same class label ratio as dataset
for train_index, test_index in skf_nb.split(train_x_resampled, train_y_resampled):
    # extract train folds x y 
    train_x = train_x_resampled[train_index]
    train_y = train_y_resampled[train_index]
    # extract test fold x and y
    test_x = train_x_resampled[test_index]
    test_y = train_y_resampled[test_index]
    
    # apply bernoulli naive bayes
    clf_nb = BernoulliNB(force_alpha=True)
    # fit model on train folds
    clf_nb = clf_nb.fit(train_x, train_y)
    # predict on test fold
    test_y_pred = clf_nb.predict(test_x)
    
    train_y_1 = 0
    train_y_0 = 0
    for label in train_y:
        if label == 1:
            train_y_1 += 1
        elif label == 0:
            train_y_0 += 1
    
    test_y_1 = 0
    test_y_0 = 0
    for label in test_y:
        if label == 1:
            test_y_1 += 1
        elif label == 0:
            test_y_0 += 1
    
    # train folds and test fold label ratio follows the ratio in the dataset which is good
    print("train label ratio: ", train_y_1 / (train_y_0 + train_y_1), train_y_0 / (train_y_0 + train_y_1))
    print("test label ratio: ", test_y_1 / (test_y_0 + test_y_1), test_y_0 / (test_y_0 + test_y_1))
    
    f1_scores = []
    print("f1 score: ", f1_score(test_y, test_y_pred))
    print("confusion matrix: \n", confusion_matrix(test_y, test_y_pred))
    f1_scores.append(f1_score(test_y, test_y_pred))

print("average f1 score: ", sum(f1_scores) / len(f1_scores))

# make prediction on validation set
label_pred = clf_nb.predict(validation_x)
labels_pred = []
for label in label_pred:
    labels_pred.append(label)
print(labels_pred)
print(validation_y)
print("f1 score validation set: ", f1_score(validation_y, labels_pred))


# In[22]:


# predict test set with nb clf
label_pred = clf_nb.predict(test_x_matrix)
for label in label_pred:
    print(label)


# #### Experiments for Perceptron Classifer with Stratified Cross Validation
# #### Different random states for experiments for reproducibility, the random_state each experiment used matches random_state in RandomOverSampler above. Details can be found in the corresponding pdf file.

# In[23]:


# # shuffle the resampled data before dividing into kfolds
# skf_per = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

# # split to n folds with the same class label ratio as dataset
# for train_index, test_index in skf_per.split(train_x_resampled, train_y_resampled):
#     # extract train folds x y 
#     train_x = train_x_resampled[train_index]
#     train_y = train_y_resampled[train_index]
#     # extract test fold x and y
#     test_x = train_x_resampled[test_index]
#     test_y = train_y_resampled[test_index]
    
#     # apply MLP classifier
#     clf_per = Perceptron(random_state=71)
#     # fit model on train fold
#     clf_per = clf_per.fit(train_x, train_y)
#     # predict on test fold
#     test_y_pred = clf_per.predict(test_x)
    
#     train_y_1 = 0
#     train_y_0 = 0
#     for label in train_y:
#         if label == 1:
#             train_y_1 += 1
#         elif label == 0:
#             train_y_0 += 1
    
#     test_y_1 = 0
#     test_y_0 = 0
#     for label in test_y:
#         if label == 1:
#             test_y_1 += 1
#         elif label == 0:
#             test_y_0 += 1
    
#     # train folds and test fold label ratio follows the ratio in the dataset which is good
#     print("train label ratio: ", train_y_1 / (train_y_0 + train_y_1), train_y_0 / (train_y_0 + train_y_1))
#     print("test label ratio: ", test_y_1 / (test_y_0 + test_y_1), test_y_0 / (test_y_0 + test_y_1))
    
#     f1_scores = []
#     print("f1 score: ", f1_score(test_y, test_y_pred))
#     print("confusion matrix: \n", confusion_matrix(test_y, test_y_pred))
#     f1_scores.append(f1_score(test_y, test_y_pred))

# print("average f1 score: ", sum(f1_scores) / len(f1_scores))

# # make prediction on validation set
# label_pred = clf_per.predict(validation_x)
# labels_pred = []
# for label in label_pred:
#     labels_pred.append(label)
# print(labels_pred)
# print(validation_y)
# print("f1 score validation set: ", f1_score(validation_y, labels_pred))


# In[24]:


# predict test set with per clf
label_pred = clf_per.predict(test_x_matrix)
for label in label_pred:
    print(label)

