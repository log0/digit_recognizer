"""

Ensembling:
1) split X into X_train & X_cv
2) with X_train, split into 'k' CV folds
3) for each 1:n models used:
for each of 'k' folds:
 use X_train[non-k] to train model[n] and predict what X_train[k] is from model[n].  Save these     predictions.  This gives you a CV prediction on the k portion of the training set for model[n]
4) you now have a matrix of X_train_rows by n_models.  that is, a matrix of several CV model predictions on the training set.
5) train a model on the set obtained in step 4.  (typically Ridge Regression, glmnet, nnls, etc... something with parameter shrinkage)
6) obtain the coefficients of the model fit in 5.  these are the weights for each model used in the ensemble.
7) fit each model 1:n on the entire X_train, predict on X_cv
8) multiply model predictions by the coefficients obtained in 6
9) sum the result.  this is your ensemble prediction.

"""

import csv
import random
import numpy as np
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cross_validation import KFold
from sklearn import preprocessing

train_file = 'data/small_train.csv'

data = [ i for i in csv.reader(file(train_file, 'rb')) ]
data = data[1:] # remove header
random.shuffle(data)

X = np.array([ i[1:] for i in data ]).astype(float)
X = preprocessing.scale(X)
Y = np.array([ i[0] for i in data ]).astype(int)

train_cutoff = len(data) * 4/5

# separate into X_train, and X_test

X_dev = X[:train_cutoff, :]
Y_dev = Y[:train_cutoff]
X_test = X[train_cutoff:, :]
Y_test = Y[train_cutoff:]

n = X_dev.shape[0]

clfs = [
    RidgeClassifier(alpha = 0.001),
    RandomForestClassifier(n_estimators = 100),    
]

blend_X = np.zeros(shape=(n, len(clfs)))

for i, clf in enumerate(clfs):
    print 'Training clf [%s]' % (i)    
    for j, (train, cv) in enumerate(KFold(n, 5)):
        X_train = X_dev[train]
        Y_train = Y_dev[train]
        X_cv = X_dev[cv]
        Y_cv = Y_dev[cv]
        
        clf = clf.fit(X_train, Y_train)
        Y_cv_predict = clf.predict(X_cv)
        
        blend_X[cv,i] = Y_cv_predict

blender = LogisticRegression()
blender.fit(blend_X, Y_dev)

blender_coef = blender.coef_

Y_test_predicts = np.zeros((X_test.shape[0], len(clfs)))
for i, clf in enumerate(clfs):
    clf.fit(X_dev, Y_dev)
    Y_test_predict = clf.predict(X_test)
    Y_test_predicts[:,i] = Y_test_predict

