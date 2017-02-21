
import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):


    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
### remove outlier
data_dict.pop("TOTAL", 0)

feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = 'total_payments'
poi = "poi"

# feature scaling for salary and exercised_stock_options
f_salary = {}
f_stock = {}
for key in data_dict:
    salary = data_dict[key]['salary']
    stock = data_dict[key]['exercised_stock_options']

    # always pay attention to NaN, in this situation, we get rid of them as we involve numerical computation
    # so given them a numerical value, like 0, will let them have a scaled feature which is not correct, and potential
    # influence the result
    if salary != 'NaN':
        f_salary[key] = salary
    if stock != 'NaN' and stock is not None:
        f_stock[key] = stock

salary_max = data_dict[max(f_salary, key=f_salary.get)]['salary']
salary_min = data_dict[min(f_salary, key=f_salary.get)]['salary']
salary_de = salary_max - salary_min
stock_max = data_dict[max(f_stock, key=f_stock.get)]['exercised_stock_options']
stock_min = data_dict[min(f_stock, key=f_stock.get)]['exercised_stock_options']
stock_de = stock_max - stock_min

for key in f_salary:
    data_dict[key]['salary'] = (f_salary[key] - salary_min)/salary_de

for key in f_stock:
    data_dict[key]['exercised_stock_options'] = (f_stock[key] - stock_min)/stock_de

features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

for f1, f2, _ in finance_features:
    plt.scatter(f1, f2)
plt.show()


kmeans = KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred = kmeans.predict(finance_features)

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
