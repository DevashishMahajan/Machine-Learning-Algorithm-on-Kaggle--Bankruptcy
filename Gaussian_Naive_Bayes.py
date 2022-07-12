# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:49:22 2022

@author: Devashish
"""
# Import necessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.metrics import roc_auc_score,accuracy_score

# Change the file path as the file path in your computer
# Read csv files from kaggle dataset as Pandas Dataframe
df=pd.read_csv("D:\CDAC ML\Cases\Bankruptcy\Bankruptcy.csv")
dum_df=pd.get_dummies(df,drop_first=True)

# X is a feature 
X=dum_df.iloc[:,2:]

#y is a label
y=dum_df.iloc[:,1]

#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2022,test_size=0.3,stratify=y)

## Gaussian Naive Bayes 
gb=GaussianNB()
gb.fit(X_train,y_train)

################## K-Folds CV ###########################
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gb=GaussianNB()
results = cross_val_score(gb,X,y,scoring='roc_auc',cv=kfold)
print(results.mean())


