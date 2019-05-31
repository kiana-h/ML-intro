#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_unix.pkl", "rb"))
print("number of people in dataset: " , len(enron_data))
print("number of features for each person: ", len(list(enron_data.values())[0]))

poiz = 0
for person , features in enron_data.items():
    if features["poi"]:
        poiz += 1

print("number of people of interest: ", poiz)

with open("../final_project/poi_names.txt") as f:
    poi_names = 0
    for line in f:
        if line.startswith("(y)") or line.startswith("(n)"):
            poi_names += 1
    print("names of people of interest: ", poi_names)
    

print("------"*10)    
print("value of james prentice stocks: " , enron_data["PRENTICE JAMES"]["total_stock_value"])
print("emails from welsey colwell to pois: " , enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("value of stock options exercised by jeffrey k skilling: " , enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])


print("------"*10)
print("\n CEO payment: " , enron_data["SKILLING JEFFREY K"]["total_payments"],"\n", 
      "Chairman payment: " , enron_data["LAY KENNETH L"]["total_payments"],"\n", 
      "CFO payment: " , enron_data["FASTOW ANDREW S"]["total_payments"],
      "\n")


salaried = 0
emails = 0
for person,features in enron_data.items():
    if not features["salary"] == "NaN":
        salaried += 1
    if not features["email_address"] == "NaN":
        emails += 1
print("------"*10)
print("people with quantified salary: " , salaried)
print("people with known emails: " , emails)

no_pay_info = 0
for person,features in enron_data.items():
    if features["total_payments"] == "NaN":
        no_pay_info += 1
print("------"*10)
print("people with no payment info: " , round(no_pay_info/len(enron_data),2))

poi_no_pay_info = 0
for person,features in enron_data.items():
    if features["total_payments"] == "NaN" and features["poi"]:
        poi_no_pay_info += 1
print("------"*10)
print("pois with no payment info: " , poi_no_pay_info)
        
