
"""
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

# feature number
feature_number = len(features_train[0])

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print(accuracy)




