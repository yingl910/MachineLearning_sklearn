
"""
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

# 'linear' kernel
#clf = SVC(kernel='linear')
# 'rbf' kernel
clf = SVC(kernel='rbf', C=10000.0)
# parameter C is a float number; larger C, higher accuracy

# smaller training set
#features_train = features_train[:len(features_train)//100] # integer
#labels_train = labels_train[:len(labels_train)//100]

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(pred, labels_test)
print(accuracy)

# extract prediction for data point
answer1 = pred[10]
answer2 = pred[26]
answer3 = pred[50]
print(answer1, answer2, answer3)

# how many elements are predicted '1'
print(sum(pred))
