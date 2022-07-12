# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:44:01 2022

@author: Devashish
"""

# Import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import tree

import warnings
warnings.filterwarnings("ignore")
# Change the file path as the file path in your computer
import os
os.chdir("D:\CDAC ML\Cases\Bankruptcy")


# Read csv files from kaggle dataset as Pandas Dataframe
df = pd.read_csv("Bankruptcy.csv")

# X is a feature 
x=df.iloc[:,2:]

#y is a label
y=df.iloc[:,1]

#### Decision Tree Classifier ####
clf = DecisionTreeClassifier(random_state=2022)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

params = {'max_depth':[3,4,None],'min_samples_split':[2,10,20],'min_samples_leaf':[1,5,10]}

gcv = GridSearchCV(clf,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)


