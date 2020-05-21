#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:04:11 2020

@author: jeffbarrecchia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split as ttSplit

train_data = pd.read_csv('~/Documents/Kaggle_Projects/titanic_train.csv')
test_data = pd.read_csv('~/Documents/Kaggle_Projects/titanic_test.csv')

# sb.countplot(y = 'Survived', hue = 'Sex', data = train_data)
# sb.swarmplot(x = 'Survived', y = 'Age', data = train_data)
# rg = ['#e74c3c', '#2ecc71']
# sb.swarmplot(x = 'Survived', y = 'Fare', data = train_data, palette = rg)
# sb.swarmplot(x = 'Survived', y = 'PassengerId', data = train_data, palette = rg)


# =============================================================================
# Drops all na value types from the dataset
# =============================================================================

train_data = train_data.dropna()
test_data = test_data.dropna()

# =============================================================================
# Drops the Name and Cabin columns from the dataset as they are not going to
# relevant to the regression
# 
# Combines the Sibling/Spouse column with the Parents on board into one column
# called TotalFamily, then drops the Sibling/Spouse and Parents columns so as
# not to have redundant data
# 
# Does this for both the train and the test datasets
# =============================================================================

train_data.drop(columns = ['Name', 'Cabin'], inplace = True)
train_data['TotalFamily'] = train_data['SibSp'] + train_data['Parch']
train_data.drop(columns = ['SibSp', 'Parch'], inplace = True)

test_data.drop(columns = ['Name', 'Cabin'], inplace = True)
test_data['TotalFamily'] = test_data['SibSp'] + test_data['Parch']
test_data.drop(columns = ['SibSp', 'Parch'], inplace = True)

# =============================================================================
# Finds the correlation between all of the variables, and constructs a heatmap
# of all of them
# =============================================================================

corr = train_data.corr()
# sb.heatmap(corr, annot=True)

# sb.pairplot(train_data)


# =============================================================================
# Sets x variable equal to every column besides the Survived column
# 
# Sets y variable equal to the Survived column
# 
# Sets cat_features_index equal to the values in the x variable where they are
# not equal to a float value
# =============================================================================

x = train_data.drop(columns = ['Survived'])
y = train_data['Survived']
cat_features_index = np.where(x.dtypes != float)[0]

# =============================================================================
# Splits the x and y set into a train/test dataset, where the training size
# is 80% of the set
# Applies a random state of 2
# =============================================================================

xtrain, xtest, ytrain, ytest = ttSplit(x, y, train_size = 0.8)

# =============================================================================
# Specifies the evaluation metric to be questioned is Accuracy, and tells the 
# program to return the best model achieved
# =============================================================================

cbc = CatBoostClassifier(eval_metric = 'Accuracy', use_best_model = True)
cbc.fit(xtrain, ytrain, cat_features = cat_features_index, eval_set=(xtest, ytest), early_stopping_rounds = 50)

# =============================================================================
# Uses the CatBoostClassifier to predict which people survive from the test
# data, where a 1 is survivie and a 0 is die
# 
# Then creates a new column in the test_data set called PredictedSurvival and
# sets it equal to the predicted deaths
# =============================================================================

prediction = cbc.predict(test_data)

test_data['PredictedSurvival'] = prediction

# =============================================================================
# Tells you how many people survived per test run
# =============================================================================

survived = 0
for i in test_data['PredictedSurvival']:
    if i == 1:
        survived += 1
print('\n\n', survived, 'people survived.')
print('\n\nMeaning that', (survived / len(test_data['PredictedSurvival'])) * 100, 'percent of passengers lived.')