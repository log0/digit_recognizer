"""
This uses RandomForest from Scikit-learn.

Average accuracy at 84.2% % on small_train.csv . (10 trees)
Average accuracy at 93.9% on train.csv . (10 trees)

"""

import csv
import random
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    train_file = 'data/train.csv'

    data = [ i for i in csv.reader(file(train_file, 'rb')) ]
    data = data[1:] # remove header
    random.shuffle(data)
    
    print 'Finish reading data'

    X = [ i[1:] for i in data ]
    Y = [ i[0] for i in data ]

    train_cutoff = len(data) * 3/4

    X_train = X[:train_cutoff]
    Y_train = Y[:train_cutoff]
    X_test = X[train_cutoff:]
    Y_test = Y[train_cutoff:]

    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    classifier = classifier.fit(X_train, Y_train)
    
    print 'Training error : %s' % (classifier.score(X_train, Y_train))

    Y_predict = classifier.predict(X_test)

    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    print 'Accuracy = %s' % (float(equal)/len(Y_predict))


