"""
This uses RandomForest with cross-validation using the parameters and pca.

Results are not good.

[k=0][n_trees=1000][n_c=50] Accuracy = 0.949166666667
[k=0][n_trees=1000][n_c=75] Accuracy = 0.94880952381
[k=1][n_trees=1000][n_c=50] Accuracy = 0.954166666667
[k=1][n_trees=1000][n_c=75] Accuracy = 0.9525
[k=2][n_trees=1000][n_c=50] Accuracy = 0.953333333333

"""

import csv
import random
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def get_accuracy(Y_predict, Y_test):
    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    return float(equal)/len(Y_predict)

if __name__ == '__main__':
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
            
        for n_components in [50, 75]:
            pca = PCA(n_components=n_components)
            pca.fit(X)
            X_rotated = pca.transform(X)
        
            X_train = X_rotated[train_index_vector]
            Y_train = Y[train_index_vector]
            X_test = X_rotated[test_index_vector]
            Y_test = Y[test_index_vector]

            for n_trees in [1000]:
                classifier = RandomForestClassifier(n_estimators=n_trees, n_jobs=3) # 3 cores
                classifier.fit(X_train, Y_train)

                Y_predict = classifier.predict(X_test)

                accuracy = get_accuracy(Y_predict, Y_test)
                """
                if accuracy > max_accuracy:
                    max_c = c
                    max_pca = pca
                    max_gamma = gamma                    
                    max_accuracy = accuracy
                    max_classifier = classifier
                """
                print '[k=%d][n_trees=%d][n_c=%d] Accuracy = %s' % (k, n_trees, n_components, accuracy)
        
            # X_submission = np.array([ [ float(j) for j in i ] for i in test_data ])
            # X_submission = max_pca.transform(X_submission)
            # Y_submission = max_classifier.predict(X_submission)

            # f = file('output/svm_cv_pca_and_predict.k%s.n_c%s.c%s.gamma%015.13f.out.csv' % (k, n_components, max_c, max_gamma), 'wb')
            # f.write('\n'.join([str(i) for i in Y_submission]))
            # f.close()
            