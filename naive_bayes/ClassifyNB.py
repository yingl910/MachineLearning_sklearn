from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    clff = GaussianNB()
    return clff.fit(features_train,labels_train)
    