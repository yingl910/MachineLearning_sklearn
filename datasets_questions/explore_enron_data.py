
""" 
    Explore the Enron dataset (emails + finances);

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
"""

import pickle

file = open("../final_project/final_project_dataset.pkl", "rb")
enron_data = pickle.load(file)

# Q1: size of Eron dataset
print(len(enron_data))

# Q2: number of features
key, value = enron_data.popitem()
print(len(value))

# Q3: identify the number of POI (['poi']==1)
counter = 0
for d in enron_data.values():
    if d['poi'] == 1:
        counter = counter + 1
print(counter)

# Q4: how many POI in the poi_name.txt
POIfile = open("../final_project/poi_names.txt", "r")
# skip the first two lines, which are not poi information
POIfile.readline()
POIfile.readline()
lines = POIfile.readlines()
print(len(lines))

# What is the total value of the stock belonging to James Prentice?
name1 = 'Prentice James'
name1 = name1.upper()
query1 = enron_data[name1]['total_stock_value']
print(query1)

# How many email messages do we have from Wesley Colwell to persons of interest?
name2 = 'Colwell Wesley'
name2 = name2.upper()
query2 = enron_data[name2]['from_this_person_to_poi']
print(query2)

#What’s the value of stock options exercised by Jeffrey K Skilling?
name3 = 'Skilling Jeffrey K'
name3 = name3.upper()
query3 = enron_data[name3]['exercised_stock_options']
print(query3)

# How many folks in this dataset have a quantified salary? What about a known email address?
counter_s = 0
counter_e = 0
for d in enron_data.values():
    if d['salary'] != 'NaN':
        counter_s = counter_s + 1

    if d['email_address'] != 'NaN':
        counter_e = counter_e + 1

print(counter_s, counter_e)

# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments?
# What percentage of people in the dataset as a whole is this?
counter_tp = 0
for d in enron_data.values():
    if d['total_payments'] == 'NaN':
        counter_tp = counter_tp + 1

print(counter_tp/len(enron_data))
