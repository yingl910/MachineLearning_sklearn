
import operator

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error = {}


    ### your code goes here
    for i in range(len(predictions)):
        error[i] = abs(predictions[i] - net_worths[i])

    sorted_error = sorted(error.items(), key=operator.itemgetter(1), reverse=True) #this is a list
    remove_index = []
    for i in range(int(len(predictions) * 0.1)):
        remove_index.append(sorted_error[i][0])

    for i in range(len(predictions)):
        if i not in remove_index:
            tup = (ages[i], net_worths[i], predictions[i] - net_worths[i])
            cleaned_data.append(tup)

    return cleaned_data

