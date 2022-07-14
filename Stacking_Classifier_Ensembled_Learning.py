# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:49:22 2022
@author: Devashish
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
import os
# Change the file path as the file path in your computer
os.chdir("D:\CDAC ML\Cases\Bankruptcy")

# Read csv files from kaggle dataset as Pandas Dataframe
df = pd.read_csv("Bankruptcy.csv")

# X is a feature
X=df.iloc[:,2:]

#y is a label
y=df.iloc[:,1]


# train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2022,stratify=y)


# estimators = SVC (linear) , SVC (rbf), GaussianNB , LogisticRegression , DecisionTreeClassifier

# estimators 1 : SVC (linear)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
svc_l = SVC(probability=True, kernel='linear',random_state=2022)
pipe_l=Pipeline([('scaler',StandardScaler()),('svc-linear',svc_l)])

# estimators 2 : SVC (rbf)
svc_r = SVC(probability=True, kernel='rbf',random_state=2022)
pipe_r=Pipeline([('scaler',StandardScaler()),('svc-rbf',svc_r)])

# estimators 3 : GaussianNB
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()

# estimators 4 : LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

# estimators 5 : DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=2022)

# Creating estimators list
models_considered=[('SVM-linear', pipe_l),('SVM-RBF',pipe_r),
                   ('Logistic Regression',logreg),
                   ('Naive Bayes',gaussian),('Decision Tree',dtc)]

#  final_estimator = Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=2022)

# Stacking Classifier
from sklearn.ensemble import StackingClassifier
stack=StackingClassifier(estimators=models_considered,
                         final_estimator=clf,
                         stack_method="predict_proba",passthrough=True)

# Fit model on training dataset
stack.fit(X_train,y_train)

# roc_auc_score on test
y_pred_proba=stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred_proba))


## Hyper parameter tuning using Grid Search Cross Validation

stack=StackingClassifier(estimators=models_considered,
                         final_estimator=clf,
                         stack_method="predict_proba",passthrough=True)


## stack.get_params()
params={ 'SVM-linear__svc-linear__C': [0.01,0.5,1,3],
        'SVM-RBF__svc-rbf__C':[0.01,0.5,1,3],
        'SVM-RBF__svc-rbf__gamma': [0.01,0.5,1,3,'scale'],
        'Decision Tree__max_depth':[None,4]}

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
gcv=GridSearchCV(estimator=stack,param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

# =============================================================================
# print(gcv.best_params_)
# {'Decision Tree__max_depth': 4, 'SVM-RBF__svc-rbf__C': 3, 'SVM-RBF__svc-rbf__gamma': 1, 'SVM-linear__svc-linear__C': 3}
# 
# print(gcv.best_score_)
# 0.9210059171597633
# =============================================================================




