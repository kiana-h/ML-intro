

     #!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from outlier_cleaner import outlierCleaner


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
data_dict.pop("TOTAL")
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
    
max_salary = 0
max_paid = None
    
for person, features in data_dict.items():
    if (features["salary"]) != "NaN":
        if int(features["salary"]) > max_salary:
            max_salary = features["salary"  ]
            max_paid  = person

print("highest salary and bonus: " , max_paid)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


