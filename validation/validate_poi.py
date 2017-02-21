
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print(accuracy)

#How many POIs are predicted for the test set for your POI identifier?
print(sum(pred))

#How many people total are in your test set?
print(len(pred))

precision_score = precision_score(labels_test, pred)
print(precision_score)

recall_score = recall_score(labels_test, pred)
print(recall_score)