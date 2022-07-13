# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:49:22 2022
@author: Devashish
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Change the file path as the file path in your computer
# Read csv files from kaggle dataset as Pandas Dataframe
df = pd.read_csv(r"D:\CDAC ML\Cases\Bankruptcy\Bankruptcy.csv")

# X is a feature
X = df.iloc[:,2:]

#y is a label
y = df.iloc[:,1]

# Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

################## Grid Search CV ###########################
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

# Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=2022)
params = {'learning_rate':np.linspace(0.001,1,5),
          'n_estimators': [100,150],
          'max_depth': [2,3,5]}
gcv = GridSearchCV(model,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X_scaled,y)
print(gcv.best_params_)
print(gcv.best_score_)


# =============================================================================
# Expected Results
# print(gcv.best_params_)
# {'learning_rate': 0.25075, 'max_depth': 3, 'n_estimators': 100}
# 
# print(gcv.best_score_)
# 0.8977176669484361
# =============================================================================
