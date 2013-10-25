"""
This uses SVM with cross-validation using the parameters (C, gamma)=(10000, 0.0000001) and (100000, 0.0000001).

Average accuracy at ??% on train.csv .

"""

import csv
import random
from sklearn import svm
import numpy as np
from sklearn.cross_validation import StratifiedKFold

def get_accuracy(Y_predict, Y_test):
    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    return float(equal)/len(Y_predict)

train_file = 'data/train.csv'
test_file = 'data/test.csv'

test_data = [ i for i in csv.reader(file(test_file, 'rb')) ]
test_data = test_data[1:] # remove header

data = [ i for i in csv.reader(file(train_file, 'rb')) ]
data = data[1:] # remove header

X = np.array([ i[1:] for i in data ])
Y = np.array([ i[0] for i in data ])
K = 5

skf = StratifiedKFold(Y, K)
for i, (train_index_vector, test_index_vector) in enumerate(skf):
    X_train = X[train_index_vector]
    Y_train = Y[train_index_vector]
    X_test = X[test_index_vector]
    Y_test = Y[test_index_vector]
    
    classifiers = []

    max_c = 0
    max_gamma = 0
    max_accuracy = 0
    max_classifier = None

    for c in [10000, 100000]:
        for gamma in [0.0000001]:
            classifier = svm.SVC(C=c, gamma=gamma)        
            classifier.fit(X_train, Y_train)

            Y_predict = classifier.predict(X_test)

            accuracy = get_accuracy(Y_predict, Y_test)
            
            if accuracy > max_accuracy:
                max_c = c
                max_gamma = gamma
                max_accuracy = accuracy
                max_classifier = classifier
            
            print '[i=%d][C=%12d][gamma=%15.13f] Accuracy = %s' % (i, c, gamma, accuracy)
            
            classifiers.append(classifier)

X_submission = [ i for i in test_data ]
Y_submission = max_classifier.predict(X_submission)

f = file('svm_and_predict.cross_validation.c%s.gamma%015.13f.out.csv' % (max_c, max_gamma), 'wb')
for i in Y_submission:
    f.write('%.6f\n' % i)
f.close()
    