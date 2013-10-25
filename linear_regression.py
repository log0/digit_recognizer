"""

"""

import csv
import random
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    train_file = 'data/small_train.csv'

    data = [ i for i in csv.reader(file(train_file, 'rb')) ]
    data = data[1:] # remove header
    random.shuffle(data)

    X = np.array([ i[1:] for i in data ]).astype(float)
    Y = np.array([ i[0] for i in data ]).astype(int)

    train_cutoff = len(data) * 3/4

    X_train = X[:train_cutoff]
    Y_train = Y[:train_cutoff]
    X_test = X[train_cutoff:]
    Y_test = Y[train_cutoff:]

    classifier = LinearRegression()
    classifier = classifier.fit(X_train, Y_train)
    
    print 'Training error : %s' % (classifier.fit(X_train, Y_train).score(X_train, Y_train))

    Y_predict = classifier.predict(X_test)

    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    print 'Accuracy = %s' % (float(equal)/len(Y_predict))


