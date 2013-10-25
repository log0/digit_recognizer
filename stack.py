"""

"""

import csv
import random
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn import cross_validation
from sklearn import metrics

def train_stack(X, Y, X_test, Y_test, n_runs = 3, k_folds = 5):
    
    ensemble = ExtraTreesClassifier(n_estimators = 50)
    
    models = [
        LogisticRegression(C = 1),
        LogisticRegression(C = 2),
        LogisticRegression(C = 3),
        RidgeClassifier(normalize = True, alpha = 1),
        RidgeClassifier(normalize = True, alpha = 2),
        RidgeClassifier(normalize = True, alpha = 0.1),
    ]
    
    Xs = []
    Ys = []
    
    for i in xrange(n_runs):
        print 'Iteration [%s/%s]' % (i+1, n_runs)
        kf = cross_validation.KFold(X.shape[0], n_folds = k_folds, indices = True, shuffle = True)
        for k, (train, cv) in enumerate(kf):
            X_train = X[train]
            Y_train = Y[train]
            X_cv = X[cv]
            Y_cv = Y[cv]
            
            Xk = []
            Yk = []
            
            for m, model in enumerate(models):
                model.fit(X_train, Y_train)
                Y_cv_pred = model.predict(X_cv)
                Xk.append(Y_cv_pred)
            
            Xk = np.column_stack(Xk)
            Yk = np.array(Y_cv)
            
            print '[k=%s] X_train.shape = %s' % (k, str(X_train.shape))
            print '[k=%s] Y_train.shape = %s' % (k, str(Y_train.shape))
            print '[k=%s] X_cv.shape = %s' % (k, str(X_cv.shape))
            print '[k=%s] Y_cv.shape = %s' % (k, str(Y_cv.shape))
            print '[k=%s] Xk.shape = %s' % (k, str(Xk.shape))
            print '[k=%s] Yk.shape = %s' % (k, str(Yk.shape))
            print ''
            
            Xs.append(Xk)
            Ys.append(Yk)
    
    blended_X_train = np.vstack(Xs)
    blended_Y_train = np.concatenate(Ys)
    
    print 'blended_X_train.shape = %s' % (str(blended_X_train.shape))
    print 'blended_Y_train.shape = %s' % (str(blended_Y_train.shape))
    
    ensemble.fit(blended_X_train, blended_Y_train)
    
    for m, model in enumerate(models):
        model.fit(X, Y)
    
    blended_X_test = []
    
    for m, model in enumerate(models):
        Y_test_pred = model.predict(X_test)
        
        blended_X_test.append(Y_test_pred)
    
    blended_X_test = np.column_stack(blended_X_test)
    blended_Y_test_pred = ensemble.predict(blended_X_test)
    
    print 'blended_X_test.shape      = %s' % (str(blended_X_test.shape))
    print 'blended_Y_test_pred.shape = %s' % (str(blended_Y_test_pred.shape))
    
    print 'Score : %s' % (metrics.accuracy_score(Y_test, blended_Y_test_pred))
        

if __name__ == '__main__':
    train_file = 'data/small_train.csv'

    data = [ i for i in csv.reader(file(train_file, 'rb')) ]
    data = data[1:] # remove header
    
    X = np.array([ i[1:] for i in data ]).astype(int)
    Y = np.array([ i[0] for i in data ]).astype(int)
    
    X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state = 1)
    
    print 'X_train.shape = %s' % (str(X_train.shape))
    print 'Y_train.shape = %s' % (str(Y_train.shape))
    print 'X_cv.shape = %s' % (str(X_cv.shape))
    print 'Y_cv.shape = %s' % (str(Y_cv.shape))
    
    train_stack(X_train, Y_train, X_cv, Y_cv)