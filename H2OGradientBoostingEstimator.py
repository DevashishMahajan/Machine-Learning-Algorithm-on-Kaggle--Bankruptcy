# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:19:52 2022

@author: Devashish
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import os
os.chdir(r"D:\CDAC ML\Cases\Bankruptcy")


# Convert csv file to dataframe and drop No column
bank = pd.read_csv("Bankruptcy.csv")
bank.drop('NO', axis = 1, inplace = True)

# Describe Features and Response Variable
X = bank.drop('D', axis = 1)
y = bank['D']

# Gradient Boosting Classifier using Grid Search CV
kfold = StratifiedKFold(n_splits = 5, random_state = 2022, shuffle = True)
model = GradientBoostingClassifier(random_state = 2022)
params = {'learning_rate': np.linspace(0.001, 1, 5),
          'n_estimators': [100, 150],
          'max_depth': [2, 3, 5]}
gcv = GridSearchCV(model, scoring = 'roc_auc', cv = kfold, param_grid = params)
gcv.fit(X, y)
print("Best Parameters: ", gcv.best_params_)
print("Best Score: ", gcv.best_score_)

h2o.init()

df = h2o.import_file("Bankruptcy.csv", destination_frame = 'Bankruptcy')

y = 'D'
X = df.col_names[2:]

# Because D is categorical
df['D'] = df['D'].asfactor()
df['D'].levels()

train, test = df.split_frame(ratios = [.7], seed = 2022)

gbm = H2OGradientBoostingEstimator(nfolds = 5,
                                        seed = 2022,
                                        keep_cross_validation_predictions = True)
gbm.train(x=X, y=y, training_frame = train, validation_frame = test, model_id = 'gbm')

print("ROC AUC Score: ", gbm.auc())
print(gbm.confusion_matrix())

gbm_params = {'learn_rate': np.linspace(0.0001, 1, 5).tolist(),
               'max_depth': [3, 5, 7],
               'ntrees': [50, 100]}
gbm = H2OGradientBoostingEstimator(distribution = "bernoulli")
gbm_grid = H2OGridSearch(model = gbm, grid_id = 'gbm_grid', hyper_params = gbm_params)
gbm_grid.train(x=X, y=y, training_frame = df, seed = 2022)

gbm_gridperf = gbm_grid.get_grid(sort_by = "auc", decreasing = True)
gbm_gridperf

best_gbm = gbm_gridperf.models[0]
best_gbm

h2o.cluster().shutdown()
