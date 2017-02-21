import numpy
import matplotlib.pyplot as plt

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg2 = LinearRegression()
reg2.fit(ages_test, net_worths_test)


# sklearn predictions are returned in an array, so you'll want to index into
# the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
# exact syntax, the point is the [0] at the end). In addition, make sure the
# argument to your prediction function is in the expected format - if you get a warning about needing a 2d array
# for your data, a list of lists will be interpreted by sklearn as such (e.g. [[27]]).
km_net_worth = reg.predict([27])[0][0]
print(km_net_worth)

#get the slope
#a 2-D array, so stick the [0][0] at the end
slope = reg.coef_[0][0]
print(slope)

# get the intercept
# a 1-D array, so stick [0] on the end
intercept = reg.intercept_[0]

# get the score on test data
test_score = reg.score(ages_test,net_worths_test)

# get the score on the training data
training_score = reg.score(ages_train,net_worths_train)
def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth":km_net_worth,
            "slope":slope,
            "intercept":intercept,
            "stats on test":test_score,
            "stats on training": training_score}