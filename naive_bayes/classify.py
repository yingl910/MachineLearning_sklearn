from sklearn.metrics import accuracy_score

def NBAccuracy(features_train, labels_train, features_test, labels_test):

    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train,labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    accuracy = accuracy_score(labels_test,pred)
    return accuracy