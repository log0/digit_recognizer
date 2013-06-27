"""
This uses SVM

Average accuracy at 97.667% on train.csv .

D:\L\source\digit_recognizer>python svm_and_predict.py
[C=       10000][gamma=0.0000000100000] Accuracy = 0.95980952381
[C=       10000][gamma=0.0000001000000] Accuracy = 0.976666666667
[C=       10000][gamma=0.0000010000000] Accuracy = 0.975904761905
[C=      100000][gamma=0.0000000100000] Accuracy = 0.95980952381
[C=      100000][gamma=0.0000001000000] Accuracy = 0.976666666667
[C=      100000][gamma=0.0000010000000] Accuracy = 0.975904761905

"""

import csv
import random
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA

train_file = 'data/train.csv'
test_file = 'data/test.csv'

data = [ i for i in csv.reader(file(train_file, 'rb')) ]
data = data[1:] # remove header
random.shuffle(data)

for n_components in [50, 75, 100]:
    X = np.array([ [ float(j) for j in i[1:] ] for i in data ])
    Y = np.array([ int(i[0]) for i in data ])
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    # print pca.explained_variance_ratio_
    X = pca.transform(X)

    train_cutoff = len(data) * 3/4

    X_train = X[:train_cutoff]
    Y_train = Y[:train_cutoff]
    X_test = X[train_cutoff:]
    Y_test = Y[train_cutoff:]

    classifiers = []

    max_c = 0
    max_gamma = 0
    max_accuracy = 0
    max_classifier = None

    for c in [100000]:
        for gamma in [0.0000001]:
            classifier = svm.SVC(C=c, gamma=gamma)
            classifier.fit(X_train, Y_train)

            Y_predict = classifier.predict(X_test)

            equal = 0
            for i in xrange(len(Y_predict)):
                if Y_predict[i] == Y_test[i]:
                    equal += 1

            accuracy = float(equal)/len(Y_predict)
            if accuracy > max_accuracy:
                max_c = c
                max_gamma = gamma
                max_accuracy = accuracy
                max_classifier = classifier
            print '[n_c=%d][C=%12d][gamma=%15.13f] Accuracy = %s' % (n_components, c, gamma, accuracy)
            
            classifiers.append(classifier)

    # test_data = [ i for i in csv.reader(file(test_file, 'rb')) ]
    # test_data = test_data[1:] # remove header
            
    # X_submission = np.array([ [ float(j) for j in i[1:] ] for i in test_data ])
    # X_submission = pca.transform(X_submission)

    # Y_submission = max_classifier.predict(X_submission)

    # f = file('svm_and_predict.c%s.gamma%015.13f.out.csv' % (max_c, max_gamma), 'wb')
    # f.write('\n'.join([str(i) for i in Y_submission]))
    # f.close()
        