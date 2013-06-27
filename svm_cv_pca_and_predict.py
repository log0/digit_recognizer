"""
This uses SVM with cross-validation using the parameters and pca.

Below results:
[k=0][n_c=75][C=       10000][gamma=0.0000001000000] Accuracy = 0.979285714286
[k=0][n_c=100][C=       10000][gamma=0.0000001000000] Accuracy = 0.97880952381
[k=1][n_c=75][C=       10000][gamma=0.0000001000000] Accuracy = 0.978928571429
[k=1][n_c=100][C=       10000][gamma=0.0000001000000] Accuracy = 0.978452380952
[k=2][n_c=75][C=       10000][gamma=0.0000001000000] Accuracy = 0.980714285714
[k=2][n_c=100][C=       10000][gamma=0.0000001000000] Accuracy = 0.979285714286
[k=3][n_c=75][C=       10000][gamma=0.0000001000000] Accuracy = 0.978452380952
[k=3][n_c=100][C=       10000][gamma=0.0000001000000] Accuracy = 0.97630952381
[k=4][n_c=75][C=       10000][gamma=0.0000001000000] Accuracy = 0.978095238095
[k=4][n_c=100][C=       10000][gamma=0.0000001000000] Accuracy = 0.979285714286

"""

import csv
import random
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

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

X = np.array([ [ float(j) for j in i[1:] ] for i in data ])
Y = np.array([ int(i[0]) for i in data ])

K = 5
skf = StratifiedKFold(Y, K)
for k, (train_index_vector, test_index_vector) in enumerate(skf):
    max_c = 0
    max_pca = None
    max_gamma = 0
    max_accuracy = 0
    max_classifier = None
        
    for n_components in [75, 100]:
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_rotated = pca.transform(X)
    
        X_train = X_rotated[train_index_vector]
        Y_train = Y[train_index_vector]
        X_test = X_rotated[test_index_vector]
        Y_test = Y[test_index_vector]
        

        for c in [10000]:
            for gamma in [0.0000001]:
                classifier = svm.SVC(C=c, gamma=gamma)        
                classifier.fit(X_train, Y_train)

                Y_predict = classifier.predict(X_test)

                accuracy = get_accuracy(Y_predict, Y_test)
                
                if accuracy > max_accuracy:
                    max_c = c
                    max_pca = pca
                    max_gamma = gamma                    
                    max_accuracy = accuracy
                    max_classifier = classifier
                
                print '[k=%d][n_c=%d][C=%12d][gamma=%15.13f] Accuracy = %s' % (k, n_components, c, gamma, accuracy)
    
        X_submission = np.array([ [ float(j) for j in i ] for i in test_data ])
        X_submission = max_pca.transform(X_submission)
        Y_submission = max_classifier.predict(X_submission)

        f = file('output/svm_cv_pca_and_predict.k%s.n_c%s.c%s.gamma%015.13f.out.csv' % (k, n_components, max_c, max_gamma), 'wb')
        f.write('\n'.join([str(i) for i in Y_submission]))
        f.close()
        