"""
This uses GradientBoostingClassifier from Scikit-learn.

Average accuracy at 81% on small_train.csv .
Average accuracy at 83% on train.csv .

"""
import csv
import random
from sklearn.ensemble import GradientBoostingClassifier

train_file = 'data/small_train.csv'

data = [ i for i in csv.reader(file(train_file, 'rb')) ]
data = data[1:] # remove header
random.shuffle(data)

X = [ i[1:] for i in data ]
Y = [ i[0] for i in data ]

train_cutoff = len(data) * 3/4

X_train = X[:train_cutoff]
Y_train = Y[:train_cutoff]
X_test = X[train_cutoff:]
Y_test = Y[train_cutoff:]

classifier = GradientBoostingClassifier(n_estimators=10)
classifier = classifier.fit(X_train, Y_train)

Y_predict = classifier.predict(X_test)

equal = 0
for i in xrange(len(Y_predict)):
    if Y_predict[i] == Y_test[i]:
        equal += 1

print 'Accuracy = %s' % (float(equal)/len(Y_predict))


