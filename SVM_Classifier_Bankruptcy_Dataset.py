"""
Created on Mon May 16 12:09:55 2022

@author: Devashish
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


df=pd.read_csv("D:\CDAC ML\Cases\Bankruptcy\Bankruptcy.csv")
dum_df=pd.get_dummies(df,drop_first=True)

X=dum_df.iloc[:,2:]
y=dum_df.iloc[:,1]



#SVM GRID SEARCH LINEAR
scaler = StandardScaler()
model = SVC(kernel='linear')
pipe = Pipeline([('scaler',scaler),('SVC',model)])
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'SVC__C':np.linspace(0.001,6,20)}
gcv = GridSearchCV(pipe,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)
print(f"\n\n SVM GCV Linear Best Param \n = {gcv.best_params_} \n\n SVM GCV Linear Best Score = \n {gcv.best_score_}")

print(gcv.best_params_)
print(gcv.best_score_)

#SVM GRID SEARCH POLYNOMIAL 
scaler = StandardScaler()
model = SVC(kernel='poly')
pipe = Pipeline([('scaler',scaler),('SVC',model)])
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'SVC__C':np.linspace(0.001,6,20),'SVC__degree':[2,3,4]}
gcv = GridSearchCV(pipe,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)
print(f"\n\n SVM GCV Polynomial Best Param \n = {gcv.best_params_} \n\n SVM GCV Polynomial Best Score = \n {gcv.best_score_}")

print(gcv.best_params_)
print(gcv.best_score_)


#SVM GRID SEARCH RADIAL
scaler = StandardScaler()

model = SVC(kernel='rbf')
pipe = Pipeline([('scaler',scaler),('SVC',model)])
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'SVC__C':np.linspace(0.001,6,20),'SVC__gamma':np.linspace(0.001,6,20)}

gcv = GridSearchCV(pipe,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)

print(f"\n\n SVM GCV Radial Best Param \n = {gcv.best_params_} \n\n SVM GCV Radial Best Score = \n {gcv.best_score_}")

print(gcv.best_params_)
print(gcv.best_score_)
